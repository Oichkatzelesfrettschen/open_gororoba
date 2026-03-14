use cudarc::driver::*;

#[test]
fn test_alloc() {
    let Ok(ctx) = CudaContext::new(0) else {
        return;
    };
    let _ = unsafe { ctx.alloc_unified::<u8>(10, false) };
}
