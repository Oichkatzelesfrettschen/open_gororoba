// Zero a single u32 atomic counter.

@group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    atomicStore(&counter[0], 0u);
}
