//! The record of the dispatches of one forward pass, with GPU timestamps when a step is
//! profiled.

use crate::device::bytes_to_words;
use crate::ops::{MatKernel, MmGroup, record_matmul};

/// What a forward pass does after the kernels.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    /// The rows of a context restart (ADR 0015): submit only. The pass stops after the last
    /// block, because the weights of these rows are not used.
    Restart,
    /// Submit and wait, no read back.
    Wait,
    /// Read the weights back.
    Read,
    /// Read the weights and the timestamps back.
    Profile,
    /// Run the coder op and read only the intervals back.
    Intervals,
}

/// The most timestamps of one pass: one for each dispatch, and the start.
pub(super) const MAX_STAMPS: u32 = 4096;

/// Records dispatches, with a timestamp after each one when profiling.
pub(super) struct Rec<'a, 'p> {
    pub(super) pass: wgpu::ComputePass<'p>,
    pub(super) qs: Option<&'a wgpu::QuerySet>,
    pub(super) labels: Vec<&'static str>,
    pub(super) mm: (&'a wgpu::ComputePipeline, &'a wgpu::ComputePipeline),
    pub(super) kernel: MatKernel,
    pub(super) n: u32,
}

impl Rec<'_, '_> {
    pub(super) fn stamp(&mut self, label: &'static str) {
        if let Some(qs) = self.qs
            && let Ok(i) = u32::try_from(self.labels.len())
            && i < MAX_STAMPS
        {
            self.pass.write_timestamp(qs, i);
            self.labels.push(label);
        }
    }

    pub(super) fn run(
        &mut self,
        label: &'static str,
        pipe: &wgpu::ComputePipeline,
        g: &wgpu::BindGroup,
        x: u32,
        y: u32,
    ) {
        dispatch(&mut self.pass, pipe, g, x, y);
        self.stamp(label);
    }

    pub(super) fn matmul(&mut self, label: &'static str, g: &MmGroup, rows_of_w: u32) {
        record_matmul(&mut self.pass, self.mm, g, self.kernel, rows_of_w, self.n);
        self.stamp(label);
    }
}

/// The ticks between each timestamp and the one before, summed for each label (first-seen
/// order).
pub(super) fn sum_stamps(labels: &[&'static str], bytes: &[u8]) -> Vec<(&'static str, u64)> {
    let words = bytes_to_words(bytes);
    let ticks: Vec<u64> = words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
        .collect();
    let mut out: Vec<(&'static str, u64)> = Vec::new();
    for (i, label) in labels.iter().enumerate().skip(1) {
        let d = ticks
            .get(i)
            .copied()
            .unwrap_or(0)
            .saturating_sub(ticks.get(i - 1).copied().unwrap_or(0));
        match out.iter_mut().find(|(l, _)| l == label) {
            Some(e) => e.1 += d,
            None => out.push((label, d)),
        }
    }
    out
}

fn dispatch(
    p: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    g: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    p.set_pipeline(pipe);
    p.set_bind_group(0, g, &[]);
    p.dispatch_workgroups(x, y, 1);
}
