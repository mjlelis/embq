use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbqError {
    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },
    #[error("Invalid data size: {size} is not a multiple of dimension {dim}")]
    InvalidDataSize { size: usize, dim: usize },
}

pub type Result<T> = std::result::Result<T, EmbqError>;

pub struct Embeddings {
    pub dim: usize,
    pub n: usize,
    pub data: Vec<f32>,
}

impl Embeddings {
    pub fn new(dim: usize, data: Vec<f32>) -> Result<Self> {
        if data.is_empty() && dim > 0 {
            return Ok(Self { dim, n: 0, data });
        }
        if !data.len().is_multiple_of(dim) {
            return Err(EmbqError::InvalidDataSize {
                size: data.len(),
                dim,
            });
        }
        let n = data.len() / dim;
        Ok(Self { dim, n, data })
    }

    pub fn normalize(&mut self) {
        for i in 0..self.n {
            let start = i * self.dim;
            let end = start + self.dim;
            let row = &mut self.data[start..end];
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in row {
                    *x /= norm;
                }
            }
        }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }
}

pub fn bruteforce_topk(db: &Embeddings, queries: &Embeddings, k: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::with_capacity(queries.n);
    for i in 0..queries.n {
        let q = queries.row(i);
        let mut scores: Vec<(usize, f32)> = (0..db.n)
            .map(|j| {
                let d = db.row(j);
                let score = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>();
                (j, score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results.push(scores.iter().take(k).map(|(idx, _)| *idx).collect());
    }
    results
}

pub struct Int8Index {
    pub dim: usize,
    pub n: usize,
    pub data: Vec<i8>,
    pub scale: f32,
}

pub fn quantize_int8(emb: &Embeddings) -> Int8Index {
    let mut amax = 0.0f32;
    for &x in &emb.data {
        amax = amax.max(x.abs());
    }
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let data = emb
        .data
        .iter()
        .map(|&x| (x / scale).round() as i8)
        .collect();
    Int8Index {
        dim: emb.dim,
        n: emb.n,
        data,
        scale,
    }
}

pub fn search_int8(idx: &Int8Index, queries: &Embeddings, k: usize) -> Vec<Vec<usize>> {
    let q_int8 = queries
        .data
        .iter()
        .map(|&x| (x / idx.scale).round() as i8)
        .collect::<Vec<_>>();
    let mut results = Vec::with_capacity(queries.n);

    for i in 0..queries.n {
        let q_start = i * idx.dim;
        let q_row = &q_int8[q_start..q_start + idx.dim];

        let mut scores: Vec<(usize, i32)> = (0..idx.n)
            .map(|j| {
                let d_start = j * idx.dim;
                let d_row = &idx.data[d_start..d_start + idx.dim];
                let score = q_row
                    .iter()
                    .zip(d_row.iter())
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum::<i32>();
                (j, score)
            })
            .collect();

        scores.sort_by_key(|b| std::cmp::Reverse(b.1));
        results.push(scores.iter().take(k).map(|(idx, _)| *idx).collect());
    }
    results
}

pub struct BinIndex {
    pub dim: usize,
    pub n: usize,
    pub data: Vec<u64>, // Packed bits
}

pub fn quantize_binary(emb: &Embeddings) -> BinIndex {
    let u64_per_row = emb.dim.div_ceil(64);
    let mut data = vec![0u64; emb.n * u64_per_row];

    for i in 0..emb.n {
        for d in 0..emb.dim {
            if emb.data[i * emb.dim + d] > 0.0 {
                let word_idx = i * u64_per_row + (d / 64);
                let bit_idx = d % 64;
                data[word_idx] |= 1 << bit_idx;
            }
        }
    }

    BinIndex {
        dim: emb.dim,
        n: emb.n,
        data,
    }
}

pub fn search_binary(idx: &BinIndex, queries: &Embeddings, k: usize) -> Vec<Vec<usize>> {
    let q_bin = quantize_binary(queries);
    let u64_per_row = idx.dim.div_ceil(64);
    let mut results = Vec::with_capacity(queries.n);

    for i in 0..queries.n {
        let q_start = i * u64_per_row;
        let q_row = &q_bin.data[q_start..q_start + u64_per_row];

        let mut scores: Vec<(usize, u32)> = (0..idx.n)
            .map(|j| {
                let d_start = j * u64_per_row;
                let d_row = &idx.data[d_start..d_start + u64_per_row];
                let dist: u32 = q_row
                    .iter()
                    .zip(d_row.iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum();
                (j, dist)
            })
            .collect();

        // Smaller Hamming distance is better
        scores.sort_by_key(|a| a.1);
        results.push(scores.iter().take(k).map(|(idx, _)| *idx).collect());
    }
    results
}

pub fn search_binary_rescore(
    idx: &BinIndex,
    db_f32: &Embeddings,
    queries: &Embeddings,
    k: usize,
    oversample: usize,
) -> Vec<Vec<usize>> {
    let q_bin = quantize_binary(queries);
    let u64_per_row = idx.dim.div_ceil(64);
    let mut results = Vec::with_capacity(queries.n);
    let candidates_to_check = k * oversample;

    for i in 0..queries.n {
        let q_start = i * u64_per_row;
        let q_row = &q_bin.data[q_start..q_start + u64_per_row];

        let mut candidates: Vec<(usize, u32)> = (0..idx.n)
            .map(|j| {
                let d_start = j * u64_per_row;
                let d_row = &idx.data[d_start..d_start + u64_per_row];
                let dist: u32 = q_row
                    .iter()
                    .zip(d_row.iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum();
                (j, dist)
            })
            .collect();

        candidates.sort_by_key(|a| a.1);

        let top_candidates = candidates.iter().take(candidates_to_check);
        let q_f32 = queries.row(i);

        let mut rescored: Vec<(usize, f32)> = top_candidates
            .map(|&(idx, _)| {
                let d_f32 = db_f32.row(idx);
                let score = q_f32
                    .iter()
                    .zip(d_f32.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                (idx, score)
            })
            .collect();

        rescored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.push(rescored.iter().take(k).map(|(idx, _)| *idx).collect());
    }
    results
}

pub fn recall_at_k(truth: &[Vec<usize>], pred: &[Vec<usize>], k: usize) -> f32 {
    if truth.is_empty() {
        return 0.0;
    }
    let total_recall: f32 = truth
        .iter()
        .zip(pred.iter())
        .map(|(t, p)| {
            let t_set: std::collections::HashSet<_> = t.iter().take(k).collect();
            let p_set: std::collections::HashSet<_> = p.iter().take(k).collect();
            t_set.intersection(&p_set).count() as f32 / k as f32
        })
        .sum();
    total_recall / truth.len() as f32
}
