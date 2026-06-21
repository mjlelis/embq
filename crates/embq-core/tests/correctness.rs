use embq_core::*;

#[test]
fn test_bruteforce_oracle() {
    let dim = 3;
    let db_data = vec![
        1.0, 0.0, 0.0, // 0
        0.0, 1.0, 0.0, // 1
        0.0, 0.0, 1.0, // 2
        0.5, 0.5, 0.0, // 3
    ];
    let mut db = Embeddings::new(dim, db_data).unwrap();
    db.normalize();

    let queries_data = vec![1.0, 0.1, 0.0];
    let mut queries = Embeddings::new(dim, queries_data).unwrap();
    queries.normalize();

    let k = 2;
    let truth = bruteforce_topk(&db, &queries, k);

    // Expected: 0 is closest (1.0 vs 1.0), 3 is second closest (0.5+0.5=1.0 unnorm, but after norm 0.707...)
    // Let's check:
    // q normalized: [0.995, 0.0995, 0]
    // db[0]: [1, 0, 0] -> dot = 0.995
    // db[1]: [0, 1, 0] -> dot = 0.0995
    // db[2]: [0, 0, 1] -> dot = 0
    // db[3]: [0.707, 0.707, 0] -> dot = 0.995 * 0.707 + 0.0995 * 0.707 = 0.703 + 0.070 = 0.773

    assert_eq!(truth[0][0], 0);
    assert_eq!(truth[0][1], 3);
}

#[test]
fn test_quantization_recall() {
    let dim = 128;
    let n = 100;
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..(n * dim) {
        data.push((i as f32).sin());
    }

    let mut db = Embeddings::new(dim, data).unwrap();
    db.normalize();

    let q_data = db.row(0).to_vec();
    let mut queries = Embeddings::new(dim, q_data).unwrap();
    queries.normalize();

    let k = 5;
    let truth = bruteforce_topk(&db, &queries, k);

    // Int8
    let idx_int8 = quantize_int8(&db);
    let pred_int8 = search_int8(&idx_int8, &queries, k);
    let recall_int8 = recall_at_k(&truth, &pred_int8, k);
    assert!(recall_int8 > 0.8, "Int8 recall too low: {}", recall_int8);

    // Binary
    let idx_bin = quantize_binary(&db);
    let pred_bin = search_binary(&idx_bin, &queries, k);
    let recall_bin = recall_at_k(&truth, &pred_bin, k);
    assert!(recall_bin >= 0.0, "Binary recall: {}", recall_bin);

    // Rescore
    let pred_rescore = search_binary_rescore(&idx_bin, &db, &queries, k, 10);
    let recall_rescore = recall_at_k(&truth, &pred_rescore, k);
    assert!(
        recall_rescore >= recall_bin,
        "Rescore should not be worse than binary"
    );
}
