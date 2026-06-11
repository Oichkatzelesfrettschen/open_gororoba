use cd_kernel::cayley_dickson::cd_multiply;
fn main() {
    let dim = 16;
    for i in 0..dim {
        for j in 0..dim {
            let mut a = vec![0.0; dim];
            a[i] = 1.0;
            let mut b = vec![0.0; dim];
            b[j] = 1.0;
            let c = cd_multiply(&a, &b);
            let k = i ^ j;
            if c[k].abs() < 0.5 {
                println!("Mismatch: {} * {} does not map to {}", i, j, k);
                return;
            }
        }
    }
    println!("XOR holds!");
}
