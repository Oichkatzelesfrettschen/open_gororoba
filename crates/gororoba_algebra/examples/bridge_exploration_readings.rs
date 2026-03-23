use gororoba_algebra::{
    ObservableReading, ObservableSignatureRegime, OkuboElement, TrialityAction,
};

fn main() {
    let reading = ObservableReading {
        regime: ObservableSignatureRegime::CompactEuclidean,
        observable: "koebisu-d2".to_string(),
        primary_value: 0.0,
        secondary_value: Some(1.0),
        flagged: true,
    };

    let a = OkuboElement::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let b = OkuboElement::new([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let c = OkuboElement::new([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let summary = a.orbit_summary(TrialityAction::CycleForward, &[a, b, c]);

    println!("{}", reading.summary_line());
    println!(
        "{}",
        reading
            .to_json_pretty()
            .expect("observable reading should serialize")
    );
    println!("{}", summary.summary_line());
    println!(
        "{}",
        summary
            .to_json_pretty()
            .expect("triality summary should serialize")
    );
}
