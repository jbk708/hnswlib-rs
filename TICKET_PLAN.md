# HNSW Performance Optimization - Implementation Tickets

This document contains a series of tickets to fix performance issues in HNSW parallel insertion, particularly the quadratic scaling caused by lock contention on reverse neighbor updates. Each ticket is self-contained and can be worked on independently.

## Prerequisites

Before starting any ticket, ensure you have:
- Rust toolchain installed (`rustc`, `cargo`)
- Git configured with your name and email
- Access to push to the repository
- Basic understanding of Rust concurrency (RwLock, Arc, Mutex)

## Common Workflow for All Tickets

### 1. Create Branch
```bash
# Navigate to repository root
cd /Users/jbk/repos/unifeb-collection/hnswlib-rs

# Fetch latest changes
git fetch origin

# Create and checkout new branch (replace TICKET_NUMBER with actual ticket number)
git checkout -b fix/TICKET_NUMBER-short-description

# Example:
# git checkout -b fix/001-serialize-entry-point-init
```

### 2. Make Changes
- Follow the ticket-specific instructions below
- Make incremental commits as you progress
- Write clear commit messages

### 3. Run Tests
```bash
# Run all tests in hnswlib-rs
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### 4. Run Linter
```bash
# Check formatting
cargo fmt --check

# Fix formatting (if needed)
cargo fmt

# Run clippy linter
cargo clippy -- -D warnings

# Fix clippy warnings (if any)
cargo clippy --fix --allow-dirty --allow-staged
```

### 5. Build Release Version
```bash
# Build release version to ensure it compiles
cargo build --release
```

### 6. Commit and Push
```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "fix: [TICKET_NUMBER] Short description

- Detailed change 1
- Detailed change 2
- Related to issue: HNSW entry point insertion hang"

# Push to remote
git push -u origin fix/TICKET_NUMBER-short-description

# Create pull request (via GitHub/GitLab UI or CLI)
```

---

## Ticket 001: Implement Batch Reverse Updates (Quadratic Scaling Fix)

**Priority**: HIGH  
**Estimated Time**: 6-8 hours  
**Difficulty**: Medium-High  
**Dependencies**: None  
**Status**: 🔄 **TODO**

### Problem

The current implementation causes **quadratic O(N²) scaling** during parallel insertion due to lock contention on reverse neighbor updates. When inserting N points:

- Each insertion connects to ~k neighbors (k = max_nb_connection, typically 48)
- Each connection requires updating the neighbor's reverse neighbor list
- The entry point becomes a neighbor to O(N) points
- All N threads compete for the entry point's write lock simultaneously
- Lock wait time grows linearly: O(N) wait × O(N) insertions = O(N²) total time

**Evidence**: Logs show lock acquisition times increasing from 64ms to 550ms as more points are inserted, with all contention on `PointId(0, 0)` (entry point).

### Solution

Implement **batch reverse updates** to eliminate lock contention:

1. **Defer updates**: Queue reverse neighbor updates instead of applying immediately during insertion
2. **Batch processing**: Group updates by target point after insertion phase completes
3. **Parallel batches**: Process different points' batches in parallel without contention

**Complexity**: Achieves O(N log N) time complexity by:
- Insertion phase: O(N log N) with no lock contention
- Batch grouping: O(N log N) to group updates by target point
- Batch processing: O(N) with one lock acquisition per point instead of O(N) per point

### Files to Modify

1. **`hnswlib-rs/src/hnsw.rs`**
   - Add batch update queue structure to `Hnsw` struct
   - Modify `reverse_update_neighborhood_simple()` to queue updates
   - Add `process_batched_reverse_updates()` method
   - Modify `parallel_insert()` to process batches after insertion
   - Modify `insert_slice()` to use batched updates

### Detailed Changes

#### Step 1: Define Update Queue Structure

**Location**: `hnswlib-rs/src/hnsw.rs`, near top of file (around line 50, after imports)

**Add**:
```rust
use hashbrown::HashMap;

/// Represents a pending reverse neighbor update
#[derive(Debug, Clone)]
struct ReverseUpdate<'b, T: Clone + Send + Sync> {
    /// Target point whose neighbors need updating
    target_point: Arc<Point<'b, T>>,
    /// New point to add to target's neighbors
    new_point: Arc<Point<'b, T>>,
    /// Distance from new_point to target_point
    distance: f32,
    /// Layer at which this connection exists
    layer: u8,
}
```

#### Step 2: Add Batch Update Queue to Hnsw Struct

**Location**: `hnswlib-rs/src/hnsw.rs`, around line 800 (in `Hnsw` struct)

**Current code**:
```rust
pub struct Hnsw<'b, T: Clone + Send + Sync, D: Distance<T> + Send + Sync> {
    max_nb_connection: usize,
    ef_construction: usize,
    extend_candidates: bool,
    keep_pruned: bool,
    max_layer: usize,
    layer_indexed_points: PointIndexation<'b, T>,
    data_dimension: usize,
    pub(crate) dist_f: D,
    pub(crate) searching: bool,
    pub(crate) datamap_opt: bool,
}
```

**Change to**:
```rust
pub struct Hnsw<'b, T: Clone + Send + Sync, D: Distance<T> + Send + Sync> {
    max_nb_connection: usize,
    ef_construction: usize,
    extend_candidates: bool,
    keep_pruned: bool,
    max_layer: usize,
    layer_indexed_points: PointIndexation<'b, T>,
    data_dimension: usize,
    pub(crate) dist_f: D,
    pub(crate) searching: bool,
    pub(crate) datamap_opt: bool,
    /// Queue for batched reverse neighbor updates
    /// Maps target point ID to list of updates for that point
    reverse_update_queue: Arc<Mutex<HashMap<PointId, Vec<ReverseUpdate<'b, T>>>>>,
}
```

#### Step 3: Initialize Queue in Constructor

**Location**: `hnswlib-rs/src/hnsw.rs`, around line 837 (in `new()` method)

**Current code**:
```rust
Hnsw {
    max_nb_connection,
    ef_construction,
    extend_candidates,
    keep_pruned,
    max_layer: adjusted_max_layer,
    layer_indexed_points,
    data_dimension: 0,
    dist_f: f,
    searching: false,
    datamap_opt: false,
}
```

**Change to**:
```rust
Hnsw {
    max_nb_connection,
    ef_construction,
    extend_candidates,
    keep_pruned,
    max_layer: adjusted_max_layer,
    layer_indexed_points,
    data_dimension: 0,
    dist_f: f,
    searching: false,
    datamap_opt: false,
    reverse_update_queue: Arc::new(Mutex::new(HashMap::new())),
}
```

#### Step 4: Modify `reverse_update_neighborhood_simple()` to Queue Updates

**Location**: `hnswlib-rs/src/hnsw.rs`, lines 1322-1398

**Replace function**:
```rust
fn reverse_update_neighborhood_simple(&self, new_point: Arc<Point<T>>) {
    trace!(
        "queueing reverse update neighbourhood for new point {:?}",
        new_point.p_id
    );
    let level = new_point.p_id.0;
    
    // Collect all neighbors that need reverse updates
    for l in (0..level + 1).rev() {
        let neighbours = new_point.neighbours.read();
        
        for q in &neighbours[l as usize] {
            if new_point.p_id == q.point_ref.p_id {
                continue; // Skip self-references
            }
            
            // Queue the update instead of applying immediately
            let update = ReverseUpdate {
                target_point: Arc::clone(&q.point_ref),
                new_point: Arc::clone(&new_point),
                distance: q.dist_to_ref,
                layer: l,
            };
            
            let mut queue = self.reverse_update_queue.lock();
            queue
                .entry(q.point_ref.p_id)
                .or_insert_with(Vec::new)
                .push(update);
        }
    }
} // end of reverse_update_neighborhood_simple
```

#### Step 5: Add Batch Processing Method

**Location**: `hnswlib-rs/src/hnsw.rs`, after `reverse_update_neighborhood_simple()` (around line 1400)

**Add new method**:
```rust
/// Process all queued reverse neighbor updates in batches
/// Groups updates by target point and processes them efficiently
fn process_batched_reverse_updates(&self) {
    debug!("Processing batched reverse updates");
    let start = std::time::Instant::now();
    
    // Extract all updates from queue
    let mut update_map: HashMap<PointId, Vec<ReverseUpdate<T>>> = {
        let mut queue = self.reverse_update_queue.lock();
        std::mem::take(&mut *queue)
    };
    
    if update_map.is_empty() {
        debug!("No reverse updates to process");
        return;
    }
    
    debug!("Processing {} target points with reverse updates", update_map.len());
    
    // Process updates for each target point
    // Sort by PointId to ensure consistent lock ordering (prevent deadlocks)
    let mut sorted_targets: Vec<_> = update_map.keys().cloned().collect();
    sorted_targets.sort();
    
    for target_id in sorted_targets {
        let updates = update_map.remove(&target_id).unwrap();
        
        // Acquire lock once for this target point
        let target_point = &updates[0].target_point;
        let mut target_neighbours = target_point.neighbours.write();
        
        // Apply all updates for this target point
        for update in updates {
            let n_to_add = PointWithOrder::<T>::new(
                &Arc::clone(&update.new_point),
                update.distance,
            );
            let l_n = n_to_add.point_ref.p_id.0 as usize;
            
            // Check if already present
            let already = target_neighbours[l_n]
                .iter()
                .position(|old| old.point_ref.p_id == update.new_point.p_id);
            
            if already.is_some() {
                continue;
            }
            
            // Add neighbor
            target_neighbours[l_n].push(Arc::new(n_to_add));
            let nbn_at_l = target_neighbours[l_n].len();
            
            // Apply shrinking if necessary
            let threshold_shrinking = if l_n > 0 {
                self.max_nb_connection
            } else {
                2 * self.max_nb_connection
            };
            
            if nbn_at_l > threshold_shrinking {
                target_neighbours[l_n].sort_unstable();
                target_neighbours[l_n].pop();
            } else {
                target_neighbours[l_n].sort_unstable();
            }
        }
    }
    
    let elapsed = start.elapsed();
    debug!(
        "Completed processing batched reverse updates in {:?}",
        elapsed
    );
} // end of process_batched_reverse_updates
```

#### Step 6: Modify `parallel_insert()` to Process Batches

**Location**: `hnswlib-rs/src/hnsw.rs`, lines 1263-1309

**Modify function** (add batch processing at end):
```rust
pub fn parallel_insert(&self, datas: &[(&Vec<T>, usize)]) {
    debug!("entering parallel_insert with {} points", datas.len());
    let start = std::time::Instant::now();

    // Insert first point sequentially to establish entry point and avoid race conditions
    if !datas.is_empty() {
        let first_point_start = std::time::Instant::now();
        self.insert((datas[0].0.as_slice(), datas[0].1));
        let first_point_duration = first_point_start.elapsed();
        debug!(
            "First point inserted sequentially in {:?}, entry point established",
            first_point_duration
        );

        // Insert remaining points in parallel
        if datas.len() > 1 {
            let parallel_start = std::time::Instant::now();
            let completed = Arc::new(AtomicUsize::new(0));
            let completed_clone = Arc::clone(&completed);
            datas[1..].par_iter().for_each(|&(item, v)| {
                self.insert((item.as_slice(), v));
                let count = completed_clone.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if count.is_multiple_of(100) {
                    info!(
                        "Inserted {} / {} points (elapsed: {:?})",
                        count + 1,
                        datas.len(),
                        parallel_start.elapsed()
                    );
                }
            });
            let parallel_duration = parallel_start.elapsed();
            info!(
                "Completed parallel insertion of {} points in {:?}",
                datas.len() - 1,
                parallel_duration
            );
        }
    }

    // Process all batched reverse updates
    self.process_batched_reverse_updates();

    let elapsed = start.elapsed();
    debug!(
        "exiting parallel_insert, completed in {:?} ({:.2}s)",
        elapsed,
        elapsed.as_secs_f64()
    );
} // end of parallel_insert
```

#### Step 7: Handle Single Insertions (Optional)

**Location**: `hnswlib-rs/src/hnsw.rs`, in `insert_slice()` method

**Option A**: Process batches immediately after single insertions (simpler, but may be slower)
**Option B**: Add a flag to control when batches are processed (more complex, better performance)

For now, **Option A** is recommended. Add after line 1256:
```rust
// For single insertions, process batches immediately
// This ensures graph consistency but may be slower
// TODO: Consider batching multiple single insertions
self.process_batched_reverse_updates();
```

### Testing

1. **Unit Test**: Verify batched updates produce same results as immediate updates
   ```rust
   #[test]
   fn test_batched_reverse_updates() {
       // Create HNSW instance
       // Insert points
       // Verify neighbor relationships are correct
       // Compare with non-batched version
   }
   ```

2. **Performance Test**: Measure insertion time improvement
   ```rust
   #[test]
   fn test_batch_update_performance() {
       // Insert 1000 points
       // Measure time with and without batching
       // Verify O(N log N) scaling
   }
   ```

### Verification Checklist

- [ ] Code compiles without errors
- [ ] All existing tests pass
- [ ] New batch update tests pass
- [ ] Performance test shows O(N log N) scaling (not O(N²))
- [ ] No lock contention warnings in logs
- [ ] Graph structure is correct (verify neighbor relationships)
- [ ] No clippy warnings
- [ ] Code is properly formatted

### Performance Expectations

- **Before**: O(N²) - insertion time grows quadratically
- **After**: O(N log N) - insertion time grows near-linearly
- **Lock contention**: Should be eliminated (no warnings in logs)
- **Memory**: O(N) additional space for update queue

### Notes

- **Graph consistency**: During insertion phase, graph is temporarily inconsistent (forward edges exist but reverse edges are queued). This is acceptable as long as:
  - No concurrent reads occur during insertion
  - Batches are processed before graph is used for queries
  
- **Periodic batch processing**: For very large datasets, consider processing batches periodically (e.g., every 1000 insertions) instead of only at the end. This reduces memory usage.

- **Backward compatibility**: This change maintains the same public API, so existing code should work without modification.

- **Edge cases to handle**:
  - Empty update queue
  - Duplicate updates (already handled with `already.is_some()` check)
  - Points that are deleted before batch processing (should not occur in current implementation)

---

## Implementation Order Recommendation

1. **Ticket 001** (Batch Reverse Updates) - **HIGH PRIORITY** - Fixes quadratic scaling, achieves O(N log N) complexity

## Testing After Implementation

After implementing Ticket 001, run comprehensive tests:

```bash
# Build release version
cargo build --release

# Run all tests
cargo test -- --nocapture

# Run performance tests to verify O(N log N) scaling
cargo test test_batch_update_performance -- --nocapture
```

## Questions or Issues?

If you encounter issues while implementing Ticket 001:
1. Check that all dependencies are installed
2. Verify you're on the correct branch
3. Run `cargo clean` and rebuild if compilation issues persist
4. Check existing tests to understand expected behavior
5. Review the analysis documents (`HNSW_QUADRATIC_SCALING_ANALYSIS.md` and `BATCH_UPDATE_COMPLEXITY_ANALYSIS.md`) for context
