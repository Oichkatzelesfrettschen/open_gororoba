pub unsafe fn read_value(pointer: *const i32) -> i32 {
    unsafe { *pointer }
}
