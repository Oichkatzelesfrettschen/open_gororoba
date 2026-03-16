use cudarc::driver::*;
use lbm_3d_cuda::sparse::{SparseBrickMap, SparseLbmSolver, SparseMemoryMode};

#[test]
fn test_alloc() {
    let Ok(ctx) = CudaContext::new(0) else {
        return;
    };
    let _ = unsafe { ctx.alloc_unified::<u8>(10, false) };
}

#[test]
fn test_sparse_managed_mode_smoke() {
    let Ok(ctx) = CudaContext::new(0) else {
        return;
    };
    let stream = ctx.default_stream();
    let mask = vec![1u8; 8 * 8 * 8];
    let d_mask = stream.clone_htod(&mask).expect("upload test mask");
    let map = SparseBrickMap::new_from_geometry(ctx.clone(), stream.clone(), 8, 8, 8, &d_mask)
        .expect("build sparse map");
    let mut solver = SparseLbmSolver::new_with_mode(map, SparseMemoryMode::ManagedUnifiedPrefetch)
        .expect("managed sparse solver");
    assert_eq!(
        solver.memory_mode(),
        SparseMemoryMode::ManagedUnifiedPrefetch
    );
    solver.evolve(1).expect("managed sparse evolve");
}
