use gororoba_structurable::{StructurableElement, structurable_v_operator_report};

fn main() {
    let x = StructurableElement::new(vec![1.0, 0.0, 0.0, 0.0]);
    let y = StructurableElement::new(vec![0.0, 1.0, 0.0, 0.0]);
    let z = StructurableElement::new(vec![0.0, 0.0, 1.0, 0.0]);

    let report = structurable_v_operator_report(&x, &y, &z);

    println!("{}", report.summary_line());
    println!(
        "{}",
        report
            .to_json_pretty()
            .expect("structurable report should serialize")
    );
}
