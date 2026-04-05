use flavor_lifts::{FlavorLift, sample_structurable_bridge};
use gororoba_structurable::StructurableElement;

struct SumLift;

impl FlavorLift for SumLift {
    fn lift(&self, v: &[f64], m: &mut faer::Mat<f64>) {
        m[(0, 0)] += v.iter().sum::<f64>();
    }
}

fn main() {
    let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
    let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
    let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

    let snapshot = sample_structurable_bridge(&x, &y, &z, &SumLift);

    println!("{}", snapshot);
    println!(
        "{}",
        snapshot
            .to_json_pretty()
            .expect("bridge snapshot should serialize")
    );
}
