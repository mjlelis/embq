// The `?` operator inside `#[pyfunction]` bodies desugars to an identity
// `From::<PyErr>::from` conversion that clippy flags as `useless_conversion`.
// The lint resolves against the macro-expanded return type, so it must be
// allowed at the crate level rather than on the function.
#![allow(clippy::useless_conversion)]

use embq_core as core;
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::time::Instant;

#[pyclass]
pub struct Report {
    #[pyo3(get)]
    pub results: Vec<MethodResult>,
}

#[pyclass]
#[derive(Clone)]
pub struct MethodResult {
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub recall_at_k: f32,
    #[pyo3(get)]
    pub latency_ms: f32,
    #[pyo3(get)]
    pub bytes_per_vec: usize,
    #[pyo3(get)]
    pub compression_ratio: f32,
}

#[pyfunction]
#[pyo3(signature = (embeddings, queries=None, k=10, methods=None, oversample=20))]
pub fn profile(
    _py: Python<'_>,
    embeddings: PyReadonlyArray2<'_, f32>,
    queries: Option<PyReadonlyArray2<'_, f32>>,
    k: usize,
    methods: Option<Vec<String>>,
    oversample: usize,
) -> PyResult<Report> {
    let emb_shape = embeddings.shape();
    let _n = emb_shape[0];
    let dim = emb_shape[1];

    let mut db =
        core::Embeddings::new(dim, embeddings.to_owned_array().into_raw_vec_and_offset().0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    db.normalize();

    let actual_queries = if let Some(q) = queries {
        let q_shape = q.shape();
        core::Embeddings::new(q_shape[1], q.to_owned_array().into_raw_vec_and_offset().0)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    } else {
        // Sample 100 queries from DB if None
        let sample_n = 100.min(db.n);
        let mut sample_data = Vec::with_capacity(sample_n * dim);
        for i in 0..sample_n {
            sample_data.extend_from_slice(db.row(i));
        }
        core::Embeddings::new(dim, sample_data).unwrap()
    };

    let mut q_norm = actual_queries;
    q_norm.normalize();

    let truth = core::bruteforce_topk(&db, &q_norm, k);
    let mut results = Vec::new();

    let methods = methods.unwrap_or_else(|| {
        vec![
            "int8".to_string(),
            "binary".to_string(),
            "binary_rescore".to_string(),
        ]
    });

    for method in methods {
        let start = Instant::now();
        let (pred, bytes_per_vec) = match method.as_str() {
            "int8" => {
                let idx = core::quantize_int8(&db);
                let p = core::search_int8(&idx, &q_norm, k);
                (p, dim) // 1 byte per dim
            }
            "binary" => {
                let idx = core::quantize_binary(&db);
                let p = core::search_binary(&idx, &q_norm, k);
                (p, dim.div_ceil(8))
            }
            "binary_rescore" => {
                let idx = core::quantize_binary(&db);
                let p = core::search_binary_rescore(&idx, &db, &q_norm, k, oversample);
                (p, dim.div_ceil(8))
            }
            _ => continue,
        };
        let latency = start.elapsed().as_secs_f32() * 1000.0 / q_norm.n as f32;
        let recall = core::recall_at_k(&truth, &pred, k);
        let fp32_bytes = dim * 4;

        results.push(MethodResult {
            method,
            recall_at_k: recall,
            latency_ms: latency,
            bytes_per_vec,
            compression_ratio: fp32_bytes as f32 / bytes_per_vec as f32,
        });
    }

    Ok(Report { results })
}

#[pymodule]
fn embq_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Report>()?;
    m.add_class::<MethodResult>()?;
    m.add_function(wrap_pyfunction!(profile, m)?)?;
    Ok(())
}
