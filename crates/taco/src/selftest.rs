//! The self-test of a GPU backend (ADR 0016): before [`crate::with_engine`] uses a device, a few
//! steps on the GPU must give the weights and the coder intervals of the CPU session, and on
//! CUDA the intervals of the mixed coder op of the profile `logs`.
//!
//! DETERMINISM: a difference is an error, so a GPU that does not give the bits of the reference
//! path is never used (CLAUDE.md §1 rule 10).

use hardshell::TokenId;
use hardshell::gpt2::{Model, Overlap, Session};
use taco_coder::{FreqBits, FreqTable, Symbol};

use crate::backend::Engine;
use crate::codec::FREQ_BITS;
use crate::error::Result;
use crate::lanes::{Answer, Lanes, Query, Stats, Table};

/// The tokens of the self-test: fixed ids spread over the vocabulary (never `<|endoftext|>`).
fn test_tokens(model: &Model) -> Vec<TokenId> {
    let vocab = u32::try_from(model.vocab_size()).unwrap_or(u32::MAX).max(1);
    (0..SELF_TEST_STEPS)
        .map(|i| {
            let t = TokenId::new((i * 7919 + 464) % vocab);
            if t == model.eot_token() { TokenId::new((t.get() + 1) % vocab) } else { t }
        })
        .collect()
}

/// The number of tokens of the self-test.
const SELF_TEST_STEPS: u32 = 6;

/// The CPU weights of the self-test: the block start, then the weights after each test token
/// except the last ([`SELF_TEST_STEPS`] rows). Cost: 6 HardShell steps on the CPU.
pub(crate) fn cpu_reference(model: &Model) -> Result<Vec<Vec<u64>>> {
    let tokens = test_tokens(model);
    let mut session = Session::new(model);
    let mut cpu_weights = vec![session.start_block()?.to_vec()];
    for &t in tokens.iter().take(tokens.len().saturating_sub(1)) {
        cpu_weights.push(session.step(t)?.to_vec());
    }
    Ok(cpu_weights)
}

/// Runs the first steps of a block on the GPU and on the CPU, and compares the weights of one
/// step and the coder answers (encode and decode) of every step. A difference is an error:
/// the GPU does not give the bits of the reference path (CLAUDE.md §1 rule 10).
pub(crate) fn self_test(
    engine: &Engine<'_>,
    model: &Model,
    cpu_weights: &[Vec<u64>],
) -> Result<()> {
    let backend = engine.backend();
    let Some(mut lanes) = engine.lanes(1, 16, Overlap::NONE)? else { return Ok(()) };
    let bits = FreqBits::new(FREQ_BITS)?;
    let tokens = test_tokens(model);
    let fail = |what: &str| backend.fault(format!("self-test: {what} differ from the CPU"));
    let mut input = model.eot_token();
    lanes.reset(0);
    for (i, (w, &t)) in cpu_weights.iter().zip(&tokens).enumerate() {
        let table = FreqTable::from_weights(w, bits)?;
        let want = table.code_interval(Symbol::new(t.get())).ok_or_else(|| fail("tokens"))?;
        // NOTE: step 0 also checks the full weights (the read back of the logs profile); the
        // later steps check the coder op, for the encoder and for the decoder.
        let got = if i == 0 {
            let gw = lanes.weights(&[(0, input)])?;
            if gw != *w {
                return Err(fail("the weights"));
            }
            Answer { token: t, interval: want }
        } else {
            let a = lanes.coded(&[(0, input)], &[Query::Encode(t)])?;
            a.first().copied().ok_or_else(|| fail("the answers"))?
        };
        if got.token != t || got.interval != want {
            return Err(fail("the encoder intervals"));
        }
        input = t;
    }
    // The decoder queries, each from a new context (`<|endoftext|>` only): the first target,
    // one inside the table, and the last target (the last symbol takes the remainder).
    let w = cpu_weights.first().ok_or_else(|| fail("the weights"))?;
    let table = FreqTable::from_weights(w, bits)?;
    for target in [0, table.total() / 3, table.total() - 1] {
        let (sym, iv) = table.symbol_at(target).ok_or_else(|| fail("targets"))?;
        lanes.reset(0);
        let a = lanes.coded(&[(0, model.eot_token())], &[Query::Decode(target)])?;
        let a = a.first().copied().ok_or_else(|| fail("the answers"))?;
        if a.token.get() != sym.get() || a.interval != iv {
            return Err(fail("the decoder intervals"));
        }
    }
    if lanes.has_mixer() {
        self_test_mixer(lanes.as_mut(), model, w, tokens.first().copied())?;
    }
    Ok(())
}

/// The self-test of the mixed coder op (the profile `logs`, ADR 0016 section 5): from a new
/// context, the dual op of the encoder and the mixed op of the decoder must give the intervals
/// of `Mixer::mix` and `FreqTable::from_weights` on the CPU, with a prediction that agrees with
/// the top token of GPT-2 and one that does not. `w` are the CPU weights of the block start.
fn self_test_mixer(
    lanes: &mut dyn Lanes,
    model: &Model,
    w: &[u64],
    token: Option<TokenId>,
) -> Result<()> {
    use taco_match::{MatchLen, Mixer, MixerParams, Prediction};
    let backend = lanes.backend();
    let fail = |what: &str| backend.fault(format!("self-test: {what} differ from the CPU"));
    let t = token.ok_or_else(|| fail("the tokens"))?;
    let bits = FreqBits::new(FREQ_BITS)?;
    let eot = model.eot_token();
    // The top token of GPT-2: the largest weight, the lowest id on a tie (as `Mixer::mix`).
    let mut top = 0usize;
    for (i, &x) in w.iter().enumerate() {
        if x > w.get(top).copied().unwrap_or(0) {
            top = i;
        }
    }
    let top = TokenId::new(u32::try_from(top).unwrap_or(0));
    let plain = FreqTable::from_weights(w, bits)?;
    let plain_iv = plain.code_interval(Symbol::new(t.get())).ok_or_else(|| fail("tokens"))?;
    for (p, len) in [(top, 9), (t, 1)] {
        let pred = Prediction {
            token: taco_match::TokenId::new(p.get()),
            length: MatchLen::new(len).ok_or_else(|| fail("the lengths"))?,
        };
        let mut mixed = w.to_vec();
        Mixer::new(MixerParams::V1).mix(&mut mixed, Some(pred))?;
        let table = FreqTable::from_weights(&mixed, bits)?;
        let want = table.code_interval(Symbol::new(t.get())).ok_or_else(|| fail("tokens"))?;
        let mut mixer = Mixer::new(MixerParams::V1);
        let mut host = |_: usize, s: Stats| {
            let agree = s.top == p;
            mixer.mix_stats(u128::from(s.total), s.predicted, agree, Some(pred)).unwrap_or(0)
        };
        lanes.reset(0);
        let a = lanes.dual(&[(0, eot)], &[t], &[Some(p)], &mut host)?;
        let (gp, gm) = a.first().copied().ok_or_else(|| fail("the answers"))?;
        if gp.token != t || gp.interval != plain_iv || gm.token != t || gm.interval != want {
            return Err(fail("the dual intervals"));
        }
        let target = table.total() / 3;
        let (sym, iv) = table.symbol_at(target).ok_or_else(|| fail("targets"))?;
        let mut mixer = Mixer::new(MixerParams::V1);
        let mut host = |_: usize, s: Stats| {
            let agree = s.top == p;
            mixer.mix_stats(u128::from(s.total), s.predicted, agree, Some(pred)).unwrap_or(0)
        };
        lanes.reset(0);
        let q = [Query::Decode(target)];
        let a = lanes.mixed(&[(0, eot)], &q, &[Table::Mixed(Some(p))], &mut host)?;
        let a = a.first().copied().ok_or_else(|| fail("the answers"))?;
        if a.token.get() != sym.get() || a.interval != iv {
            return Err(fail("the mixed decoder intervals"));
        }
    }
    Ok(())
}
