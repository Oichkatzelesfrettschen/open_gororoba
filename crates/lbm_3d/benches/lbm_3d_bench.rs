use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| black_box(2 + 2)));
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
