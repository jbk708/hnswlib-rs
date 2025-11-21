#![allow(clippy::needless_range_loop)]
#![allow(clippy::range_zip_with_len)]

use anndists::dist::*;
use hnsw_rs::prelude::*;
use rand::{distr::Uniform, Rng};

/// Test that batched reverse updates produce correct neighbor relationships
#[test]
fn test_batched_reverse_updates() {
    let nb_elem = 100;
    let max_nb_connection = 16;
    let nb_layer = 16;
    let ef_c = 200;

    // Create HNSW instance
    let mut hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});

    // Generate random data points
    let mut rng = rand::rng();
    let mut data: Vec<(Vec<f32>, usize)> = Vec::new();
    for i in 0..nb_elem {
        let point: Vec<f32> = (0..10).map(|_| rng.random_range(0.0..1.0)).collect();
        data.push((point, i));
    }

    // Insert points using parallel_insert (which uses batched updates)
    let data_for_par_insertion: Vec<(&Vec<f32>, usize)> =
        data.iter().map(|(v, id)| (v, *id)).collect();
    hnsw.parallel_insert(&data_for_par_insertion);

    // Verify that neighbor relationships are bidirectional
    // For each point, check that if A has B as a neighbor, then B has A as a neighbor
    let layer_indexed_points = hnsw.get_point_indexation();

    // Check a sample of points to verify correctness
    for i in 0..(nb_elem.min(20)) {
        if let Some(point) = layer_indexed_points.get_point_by_origin_id(i) {
            let neighbors = point.get_neighborhood_id();

            // For each layer
            for (layer, layer_neighbors) in neighbors.iter().enumerate() {
                // For each neighbor at this layer
                for neighbor in layer_neighbors {
                    let neighbor_id = neighbor.d_id;

                    // Get the neighbor point
                    if let Some(neighbor_point) =
                        layer_indexed_points.get_point_by_origin_id(neighbor_id)
                    {
                        let neighbor_neighbors = neighbor_point.get_neighborhood_id();

                        // Check if the original point is in the neighbor's neighbor list at this layer
                        let found = neighbor_neighbors[layer].iter().any(|n| n.d_id == i);

                        // The relationship should be bidirectional
                        assert!(
                            found,
                            "Point {} has neighbor {} at layer {}, but {} doesn't have {} as neighbor",
                            i, neighbor_id, layer, neighbor_id, i
                        );
                    }
                }
            }
        }
    }
}

/// Test that batched updates work correctly with single insertions
#[test]
fn test_single_insertion_batched_updates() {
    let nb_elem = 50;
    let max_nb_connection = 16;
    let nb_layer = 16;
    let ef_c = 200;

    // Create HNSW instance
    let mut hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});

    // Generate random data points
    let mut rng = rand::rng();
    let mut data: Vec<(Vec<f32>, usize)> = Vec::new();
    for i in 0..nb_elem {
        let point: Vec<f32> = (0..10).map(|_| rng.random_range(0.0..1.0)).collect();
        data.push((point, i));
    }

    // Insert points one by one (which should also use batched updates)
    for (point, id) in &data {
        hnsw.insert((point.as_slice(), *id));
    }

    // Verify that neighbor relationships are bidirectional
    let layer_indexed_points = hnsw.get_point_indexation();

    // Check a sample of points
    for i in 0..(nb_elem.min(10)) {
        if let Some(point) = layer_indexed_points.get_point_by_origin_id(i) {
            let neighbors = point.get_neighborhood_id();

            for (layer, layer_neighbors) in neighbors.iter().enumerate() {
                for neighbor in layer_neighbors {
                    let neighbor_id = neighbor.d_id;

                    if let Some(neighbor_point) =
                        layer_indexed_points.get_point_by_origin_id(neighbor_id)
                    {
                        let neighbor_neighbors = neighbor_point.get_neighborhood_id();

                        let found = neighbor_neighbors[layer].iter().any(|n| n.d_id == i);

                        assert!(
                            found,
                            "Point {} has neighbor {} at layer {}, but {} doesn't have {} as neighbor",
                            i, neighbor_id, layer, neighbor_id, i
                        );
                    }
                }
            }
        }
    }
}

/// Performance test to verify O(N log N) scaling (not O(N²))
/// This test measures insertion time and checks that it doesn't grow quadratically
#[test]
fn test_batch_update_performance() {
    let max_nb_connection = 16;
    let nb_layer = 16;
    let ef_c = 200;

    let mut rng = rand::rng();

    // Test with different sizes
    let sizes = vec![100, 200, 400];
    let mut times = Vec::new();

    for &nb_elem in &sizes {
        let mut hnsw =
            Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});

        // Generate random data
        let mut data: Vec<(Vec<f32>, usize)> = Vec::new();
        for i in 0..nb_elem {
            let point: Vec<f32> = (0..10).map(|_| rng.random_range(0.0..1.0)).collect();
            data.push((point, i));
        }

        let data_for_par_insertion: Vec<(&Vec<f32>, usize)> =
            data.iter().map(|(v, id)| (v, *id)).collect();

        // Measure insertion time
        let start = std::time::Instant::now();
        hnsw.parallel_insert(&data_for_par_insertion);
        let elapsed = start.elapsed();

        times.push((nb_elem, elapsed.as_secs_f64()));
    }

    // Verify that time doesn't grow quadratically
    // If it were O(N²), doubling size would quadruple time
    // With O(N log N), doubling size should roughly double time (or slightly more)
    if times.len() >= 2 {
        let (size1, time1) = times[0];
        let (size2, time2) = times[1];

        let ratio = time2 / time1;
        let size_ratio = size2 as f64 / size1 as f64;

        // For O(N log N), ratio should be roughly size_ratio * log(size_ratio) / log(2)
        // For O(N²), ratio would be size_ratio²
        // We check that ratio is less than size_ratio² (quadratic scaling)
        let quadratic_ratio = size_ratio * size_ratio;

        assert!(
            ratio < quadratic_ratio,
            "Insertion time ratio {} suggests worse than O(N log N) scaling (quadratic would be {})",
            ratio, quadratic_ratio
        );
    }
}
