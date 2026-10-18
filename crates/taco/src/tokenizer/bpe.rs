//! The BPE merge loop for one piece: the same result as `_bpe` in `tools/baseline/bpe.py`, in
//! `O(n log n)` instead of `O(n^2)`.
//!
//! The reference repeats these steps until no adjacent pair has a merge:
//!
//! 1. Find the lowest merge rank `r` of all adjacent pairs.
//! 2. Scan the symbols from left to right. Join each occurrence of the pair of rank `r`. After
//!    a join, the scan continues after the pair, so `a a a` gives `aa a`.
//!
//! This module does one "round" for each step. A min-heap holds every adjacent pair that has
//! a merge, keyed by `(rank, position)`. A round takes all entries of the lowest rank, sorts
//! them by position, and joins each one that is still valid. An entry is not valid when an
//! earlier join of the round used one of its symbols: this is the skip of step 2. A join
//! makes at most two new pairs. They go into the heap for later rounds. They cannot have the
//! rank `r`, because the joined token has more bytes than either part. Thus each round is
//! exactly one step of the reference.
//!
//! NOTE: a simple heap without rounds gives a different result when a new pair has a lower
//! rank than the current round. GPT-2 merges never do this, but the tokenizer takes any merge
//! list, so the rounds are necessary.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// A merge: its rank and the id of the joined token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Merge {
    pub(super) rank: u32,
    pub(super) merged: u32,
}

/// The merges, keyed by the pair `(left id, right id)`.
///
/// DETERMINISM: the map is used only for look-ups, never for iteration, so its random state
/// cannot change a result.
pub(super) type MergeMap = HashMap<(u32, u32), Merge>;

/// "No node": the end of the linked list.
const NONE: usize = usize::MAX;

/// Applies the merges to the symbols of one piece, and appends the result to `out`.
///
/// `symbols` are the single-byte token ids of the piece. Cost: `O(n log n)` for `n` symbols,
/// with 3 words of memory for each symbol.
pub(super) fn merge_into(symbols: &[u32], merges: &MergeMap, out: &mut Vec<u32>) {
    if symbols.len() < 2 {
        out.extend_from_slice(symbols);
        return;
    }
    let n = symbols.len();
    let mut sym = symbols.to_vec();
    let mut next: Vec<usize> = (1..=n).map(|i| if i == n { NONE } else { i }).collect();
    let mut prev: Vec<usize> = (0..n).map(|i| i.checked_sub(1).unwrap_or(NONE)).collect();
    let mut alive = vec![true; n];
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let rank_at = |sym: &[u32], next: &[usize], i: usize| -> Option<Merge> {
        let j = *next.get(i)?;
        merges.get(&(*sym.get(i)?, *sym.get(j)?)).copied()
    };
    for i in 0..n - 1 {
        if let Some(m) = rank_at(&sym, &next, i) {
            heap.push(Reverse((m.rank, i)));
        }
    }
    let mut batch: Vec<usize> = Vec::new();
    while let Some(&Reverse((rank, _))) = heap.peek() {
        batch.clear();
        while let Some(&Reverse((r, i))) = heap.peek() {
            if r != rank {
                break;
            }
            heap.pop();
            batch.push(i);
        }
        batch.sort_unstable();
        batch.dedup();
        for &i in &batch {
            if !alive.get(i).copied().unwrap_or(false) {
                continue;
            }
            let Some(m) = rank_at(&sym, &next, i) else { continue };
            if m.rank != rank {
                continue;
            }
            // Join node i and its next node j into node i.
            let j = next.get(i).copied().unwrap_or(NONE);
            let after = next.get(j).copied().unwrap_or(NONE);
            if let (Some(s), Some(a), Some(nx)) =
                (sym.get_mut(i), alive.get_mut(j), next.get_mut(i))
            {
                *s = m.merged;
                *a = false;
                *nx = after;
            }
            if let Some(p) = prev.get_mut(after) {
                *p = i;
            }
            let before = prev.get(i).copied().unwrap_or(NONE);
            for k in [before, i] {
                if k != NONE
                    && let Some(m) = rank_at(&sym, &next, k)
                {
                    heap.push(Reverse((m.rank, k)));
                }
            }
        }
    }
    let mut i = 0;
    while i != NONE {
        if let Some(&s) = sym.get(i) {
            out.push(s);
        }
        i = next.get(i).copied().unwrap_or(NONE);
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    /// The reference algorithm of `tools/baseline/bpe.py`, on ids.
    fn reference(symbols: &[u32], merges: &MergeMap) -> Vec<u32> {
        let mut word = symbols.to_vec();
        while word.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..word.len() - 1 {
                if let Some(m) = merges.get(&(word[i], word[i + 1]))
                    && best.is_none_or(|(r, _)| m.rank < r)
                {
                    best = Some((m.rank, i));
                }
            }
            let Some((_, bi)) = best else { break };
            let (first, second) = (word[bi], word[bi + 1]);
            let merged = merges[&(first, second)].merged;
            let mut out = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    out.push(merged);
                    i += 2;
                } else {
                    out.push(word[i]);
                    i += 1;
                }
            }
            word = out;
        }
        word
    }

    fn map(list: &[((u32, u32), u32)]) -> MergeMap {
        list.iter()
            .enumerate()
            .map(|(rank, &(pair, merged))| (pair, Merge { rank: rank as u32, merged }))
            .collect()
    }

    fn run(symbols: &[u32], merges: &MergeMap) -> Vec<u32> {
        let mut out = Vec::new();
        merge_into(symbols, merges, &mut out);
        out
    }

    #[test]
    fn overlapping_pairs_join_from_the_left() {
        // a = 0, aa = 10.
        let m = map(&[((0, 0), 10)]);
        assert_eq!(run(&[0, 0, 0], &m), [10, 0]);
        assert_eq!(run(&[0, 0, 0, 0], &m), [10, 10]);
    }

    #[test]
    fn a_new_pair_with_a_lower_rank_waits_for_the_next_round() {
        // a = 0, b = 1. Rank 0: (ab, a) -> aba = 11. Rank 1: (a, b) -> ab = 10.
        let m = map(&[((10, 0), 11), ((0, 1), 10)]);
        let input = [0, 1, 0, 1];
        assert_eq!(run(&input, &m), reference(&input, &m));
        assert_eq!(run(&input, &m), [10, 10]);
    }

    #[test]
    fn matches_the_reference_on_random_merges() {
        // A small alphabet and random merge lists, with a seeded integer generator.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut rand = move |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        for _ in 0..300 {
            let mut list = Vec::new();
            let mut next_id = 4u32;
            for _ in 0..rand(12) {
                let pair = (rand(u64::from(next_id)) as u32, rand(u64::from(next_id)) as u32);
                if list.iter().any(|&(p, _)| p == pair) {
                    continue;
                }
                list.push((pair, next_id));
                next_id += 1;
            }
            // Shuffle the ranks, so that later tokens can have lower ranks.
            for i in (1..list.len()).rev() {
                list.swap(i, rand(i as u64 + 1) as usize);
            }
            let m = map(&list);
            let len = rand(24) as usize;
            let input: Vec<u32> = (0..len).map(|_| rand(4) as u32).collect();
            assert_eq!(run(&input, &m), reference(&input, &m), "{list:?} {input:?}");
        }
    }
}
