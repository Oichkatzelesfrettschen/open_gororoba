//! Sequential Rust driver for the retained charged-hadron Fortran interpolator.
//!
//! The default-real reference preserves the retained source's literal semantics.
//! A default-real-8 build changes both grid knots and the heavy/gluon exponent;
//! its outputs are a precision amendment, rather than a knot-only experiment.
//! Pinned mirror conformance establishes neither author-release authentication
//! nor physical fragmentation calibration.

unsafe extern "C" {
    static mut fragini_: i32;
    fn fdss_(
        hadron: *const i32,
        charge: *const i32,
        order: *const i32,
        fraction: *const f64,
        scale_squared: *const f64,
        up: *mut f64,
        antiup: *mut f64,
        down: *mut f64,
        antidown: *mut f64,
        strange: *mut f64,
        antistrange: *mut f64,
        charm: *mut f64,
        bottom: *mut f64,
        gluon: *mut f64,
    );
}

fn evaluate(order: i32, charge: i32, fraction: f64, scale_squared: f64) -> [f64; 9] {
    let mut values = [0.0; 9];
    let output = values.as_mut_ptr();
    // The Fortran routine accepts scalar references and mutates shared saved state.
    // Every call is sequential and each output pointer addresses a distinct element.
    unsafe {
        fdss_(
            &4,
            &charge,
            &order,
            &fraction,
            &scale_squared,
            output,
            output.add(1),
            output.add(2),
            output.add(3),
            output.add(4),
            output.add(5),
            output.add(6),
            output.add(7),
            output.add(8),
        );
    }
    assert!(values.iter().all(|value| value.is_finite()));
    values
}

fn emit(order: i32, charge: &str, fraction: f64, scale_squared: f64, values: [f64; 9]) {
    print!("{order},{charge},{fraction:.17e},{scale_squared:.17e}");
    for value in values {
        print!(",{value:.17e}");
    }
    println!();
}

fn main() {
    let source_knots: [f64; 35] = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095, 0.1, 0.125, 0.15, 0.175, 0.2,
        0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.93, 1.0,
    ];
    eprintln!(
        "literal_bits_method=Rust_decimal64_and_binary32_promotion; compiler_default_real_semantics_declared_separately"
    );
    for (index, knot) in source_knots.into_iter().enumerate() {
        eprintln!(
            "knot={index} decimal64={knot:.17e} decimal64_bits={:016x} promoted_binary32_bits={:016x}",
            knot.to_bits(),
            f64::from(knot as f32).to_bits(),
        );
    }
    eprintln!(
        "exponent_decimal64_bits={:016x} exponent_promoted_binary32_bits={:016x}",
        0.3_f64.to_bits(),
        f64::from(0.3_f32).to_bits(),
    );
    let fractions = [
        0.05, 0.095, 0.1, 0.2, 0.225, 0.35, 0.5, 0.7, 0.93, 0.999, 1.0, 0.053, 0.137, 0.333, 0.777,
        0.975,
    ];
    let scales_squared = [1.0, 1.25, 10.0, 100.0, 1e5, 1.1, 37.0, 1777.0, 99999.0];
    println!("order,charge,z,q2,u,ub,d,db,s,sb,c,b,g");
    for order in [0, 1] {
        // The retained COMMON flag must reset before loading a different order.
        unsafe { fragini_ = 0 };
        for fraction in fractions {
            for scale_squared in scales_squared {
                let positive = evaluate(order, 1, fraction, scale_squared);
                let negative = evaluate(order, -1, fraction, scale_squared);
                let average = evaluate(order, 0, fraction, scale_squared);
                let sum = std::array::from_fn(|index| positive[index] + negative[index]);
                emit(order, "positive", fraction, scale_squared, positive);
                emit(order, "negative", fraction, scale_squared, negative);
                emit(order, "average", fraction, scale_squared, average);
                emit(order, "sum", fraction, scale_squared, sum);
            }
        }
    }
}
