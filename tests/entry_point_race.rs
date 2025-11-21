//! Tests for concurrent entry point initialization in HNSW.
//!
//! These tests verify that multiple threads can safely insert points into an HNSW
//! graph concurrently without hanging or causing race conditions when initializing
//! the entry point.

use anndists::dist::DistL2;
use hnsw_rs::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Generate a simple test vector with a unique value at index 0
fn gen_test_vector(dim: usize, id: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    vec[0] = id as f32;
    vec
}

#[test]
fn test_concurrent_entry_point_initialization() {
    const NUM_POINTS: usize = 100;
    const NUM_THREADS: usize = 10;
    const POINTS_PER_THREAD: usize = NUM_POINTS / NUM_THREADS;
    const DIM: usize = 10;

    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(
        16, // max_conn
        NUM_POINTS,
        4,   // max_layer
        200, // ef_construction
        DistL2 {},
    ));

    // Prepare test data
    let mut all_data = Vec::new();
    for i in 0..NUM_POINTS {
        let vec = gen_test_vector(DIM, i);
        all_data.push((vec, i));
    }

    // Spawn threads to insert points concurrently
    let start = Instant::now();
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let hnsw_clone = Arc::clone(&hnsw);
            let thread_data: Vec<_> = all_data
                [thread_id * POINTS_PER_THREAD..(thread_id + 1) * POINTS_PER_THREAD]
                .iter()
                .map(|(v, id)| (v.clone(), *id))
                .collect();

            thread::spawn(move || {
                for (vec, id) in thread_data {
                    hnsw_clone.insert((vec.as_slice(), id));
                }
            })
        })
        .collect();

    // Wait for all threads with timeout
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }

    let elapsed = start.elapsed();

    // Verify entry point was set
    assert!(
        hnsw.get_nb_point() > 0,
        "At least one point should have been inserted"
    );

    // Verify no hang (should complete quickly)
    assert!(
        elapsed < Duration::from_secs(30),
        "Insertion should complete within 30 seconds, took {:?}",
        elapsed
    );

    println!(
        "Successfully inserted {} points concurrently in {:?}",
        hnsw.get_nb_point(),
        elapsed
    );
}

#[test]
fn test_parallel_insert_entry_point_race() {
    const NUM_POINTS: usize = 1000;
    const DIM: usize = 10;

    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(16, NUM_POINTS, 4, 200, DistL2 {}));

    // Prepare test data
    let mut all_data = Vec::new();
    for i in 0..NUM_POINTS {
        let vec = gen_test_vector(DIM, i);
        all_data.push((vec, i));
    }

    let data_refs: Vec<_> = all_data.iter().map(|(v, id)| (v, *id)).collect();

    // Insert all points in parallel
    let start = Instant::now();
    hnsw.parallel_insert(&data_refs);
    let elapsed = start.elapsed();

    // Verify all points were inserted
    assert_eq!(
        hnsw.get_nb_point(),
        NUM_POINTS,
        "All points should have been inserted"
    );

    // Verify no hang
    assert!(
        elapsed < Duration::from_secs(60),
        "Parallel insertion should complete within 60 seconds, took {:?}",
        elapsed
    );

    println!(
        "Successfully inserted {} points in parallel in {:?}",
        NUM_POINTS, elapsed
    );
}

#[test]
fn test_single_point_insertion() {
    const DIM: usize = 10;

    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(16, 1, 4, 200, DistL2 {}));

    let vec = gen_test_vector(DIM, 0);
    let start = Instant::now();
    hnsw.insert((vec.as_slice(), 0));
    let elapsed = start.elapsed();

    // Verify point was inserted
    assert_eq!(
        hnsw.get_nb_point(),
        1,
        "Single point should have been inserted"
    );

    // Should complete very quickly
    assert!(
        elapsed < Duration::from_secs(1),
        "Single point insertion should complete within 1 second, took {:?}",
        elapsed
    );

    println!("Successfully inserted single point in {:?}", elapsed);
}

#[test]
fn test_empty_data_insertion() {
    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(16, 0, 4, 200, DistL2 {}));

    let data_refs: Vec<(&Vec<f32>, usize)> = Vec::new();
    let start = Instant::now();
    hnsw.parallel_insert(&data_refs);
    let elapsed = start.elapsed();

    // Should handle empty data gracefully
    assert_eq!(
        hnsw.get_nb_point(),
        0,
        "No points should be inserted for empty data"
    );

    // Should complete very quickly
    assert!(
        elapsed < Duration::from_secs(1),
        "Empty data insertion should complete within 1 second, took {:?}",
        elapsed
    );

    println!("Successfully handled empty data insertion in {:?}", elapsed);
}

#[test]
fn test_rapid_concurrent_insertions() {
    const NUM_POINTS: usize = 50;
    const NUM_THREADS: usize = 20; // More threads than points to stress test
    const DIM: usize = 10;

    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(16, NUM_POINTS, 4, 200, DistL2 {}));

    // Prepare test data
    let mut all_data = Vec::new();
    for i in 0..NUM_POINTS {
        let vec = gen_test_vector(DIM, i);
        all_data.push((vec, i));
    }

    // Spawn many threads, each inserting one point
    let start = Instant::now();
    let handles: Vec<_> = (0..NUM_POINTS.min(NUM_THREADS))
        .map(|i| {
            let hnsw_clone = Arc::clone(&hnsw);
            let vec = all_data[i].0.clone();
            let id = all_data[i].1;

            thread::spawn(move || {
                hnsw_clone.insert((vec.as_slice(), id));
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }

    let elapsed = start.elapsed();

    // Verify points were inserted
    assert!(
        hnsw.get_nb_point() > 0,
        "At least one point should have been inserted"
    );

    // Verify no hang
    assert!(
        elapsed < Duration::from_secs(30),
        "Rapid concurrent insertions should complete within 30 seconds, took {:?}",
        elapsed
    );

    println!(
        "Successfully completed rapid concurrent insertions: {} points in {:?}",
        hnsw.get_nb_point(),
        elapsed
    );
}

#[test]
fn test_parallel_insert_sequential_first_point() {
    const NUM_POINTS: usize = 100;
    const DIM: usize = 10;

    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(16, NUM_POINTS, 4, 200, DistL2 {}));

    // Prepare test data
    let mut all_data = Vec::new();
    for i in 0..NUM_POINTS {
        let vec = gen_test_vector(DIM, i);
        all_data.push((vec, i));
    }

    let data_refs: Vec<_> = all_data.iter().map(|(v, id)| (v, *id)).collect();

    // Insert all points using parallel_insert
    // The first point should be inserted sequentially, establishing the entry point
    // before parallel insertion of remaining points begins
    let start = Instant::now();
    hnsw.parallel_insert(&data_refs);
    let elapsed = start.elapsed();

    // Verify all points were inserted
    assert_eq!(
        hnsw.get_nb_point(),
        NUM_POINTS,
        "All points should have been inserted via parallel_insert"
    );

    // Verify no hang - should complete quickly
    assert!(
        elapsed < Duration::from_secs(60),
        "Parallel insertion with sequential first point should complete within 60 seconds, took {:?}",
        elapsed
    );

    // Verify we can perform searches (entry point must be established)
    let query_vec = gen_test_vector(DIM, 0);
    let results = hnsw.search(&query_vec, 5, 10);
    assert!(
        !results.is_empty(),
        "Search should return results if entry point was established correctly"
    );

    println!(
        "Successfully inserted {} points with sequential first point in {:?}, search returned {} results",
        NUM_POINTS,
        elapsed,
        results.len()
    );
}

#[test]
fn test_parallel_insert_single_point() {
    const DIM: usize = 10;

    let hnsw = Arc::new(Hnsw::<f32, DistL2>::new(16, 1, 4, 200, DistL2 {}));

    let vec = gen_test_vector(DIM, 0);
    let data_refs = vec![(&vec, 0)];

    // Insert single point using parallel_insert
    // This should work correctly even though there's only one point
    let start = Instant::now();
    hnsw.parallel_insert(&data_refs);
    let elapsed = start.elapsed();

    // Verify point was inserted
    assert_eq!(
        hnsw.get_nb_point(),
        1,
        "Single point should have been inserted via parallel_insert"
    );

    // Should complete very quickly
    assert!(
        elapsed < Duration::from_secs(1),
        "Single point parallel_insert should complete within 1 second, took {:?}",
        elapsed
    );

    // Verify we can perform a search (entry point must be established)
    let query_vec = gen_test_vector(DIM, 0);
    let results = hnsw.search(&query_vec, 1, 10);
    assert_eq!(
        results.len(),
        1,
        "Search should return the single inserted point"
    );

    println!(
        "Successfully inserted single point via parallel_insert in {:?}",
        elapsed
    );
}
