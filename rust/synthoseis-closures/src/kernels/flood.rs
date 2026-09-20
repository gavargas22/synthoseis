//! Priority-flood depression fill.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone)]
struct HeapItem {
    h: f64,
    row: usize,
    col: usize,
    edge: bool,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.h == other.h
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // min-heap via reverse
        other.h.partial_cmp(&self.h).unwrap_or(Ordering::Equal)
    }
}

/// 2-D priority-flood depression fill (border pixels seed the queue).
///
/// Port of the core loop in `Closures.flood_fill_heap` with a simplified edge
/// mask (array border + empty cells) — no scipy binary erosion.
/// `shape = [nrows, ncols]`, row-major. Empty cells use `empty_value`.
pub fn flood_fill_heap_2d(input: &[f64], shape: [usize; 2], empty_value: f64) -> Vec<f64> {
    let [nr, nc] = shape;
    assert_eq!(input.len(), nr * nc);
    let mut input_array = input.to_vec();
    for v in &mut input_array {
        if v.is_nan() {
            *v = empty_value;
        }
    }
    let mut max_v = f64::NEG_INFINITY;
    for &v in &input_array {
        if v > max_v {
            max_v = v;
        }
    }
    let h_max = max_v * 2.0;
    let mut inside = vec![true; nr * nc];
    for c in 0..nc {
        inside[c] = false;
        inside[(nr - 1) * nc + c] = false;
    }
    for r in 0..nr {
        inside[r * nc] = false;
        inside[r * nc + (nc - 1)] = false;
    }
    for i in 0..(nr * nc) {
        if input_array[i] >= empty_value / 2.0 {
            inside[i] = false;
        }
    }
    let mut output = input_array.clone();
    for i in 0..(nr * nc) {
        if inside[i] {
            output[i] = h_max;
        }
    }
    let mut heap = BinaryHeap::new();
    for r in 0..nr {
        for c in 0..nc {
            let i = r * nc + c;
            if !inside[i] {
                heap.push(HeapItem {
                    h: output[i],
                    row: r,
                    col: c,
                    edge: true,
                });
            }
        }
    }
    let neighbors = [(-1isize, 0), (1, 0), (0, -1), (0, 1)];
    while let Some(item) = heap.pop() {
        for &(dr, dc) in &neighbors {
            let nr_i = item.row as isize + dr;
            let nc_i = item.col as isize + dc;
            if nr_i < 0 || nc_i < 0 || nr_i as usize >= nr || nc_i as usize >= nc {
                continue;
            }
            let n_row = nr_i as usize;
            let n_col = nc_i as usize;
            let ni = n_row * nc + n_col;
            if item.edge && !inside[ni] {
                continue;
            }
            if output[ni] == h_max {
                output[ni] = item.h.max(input_array[ni]);
                heap.push(HeapItem {
                    h: output[ni],
                    row: n_row,
                    col: n_col,
                    edge: false,
                });
            }
        }
    }
    for v in &mut output {
        if (*v - empty_value).abs() < 1e-12 {
            *v = f64::NAN;
        }
    }
    output
}
