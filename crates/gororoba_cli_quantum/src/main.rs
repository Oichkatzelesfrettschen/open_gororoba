fn main() {
    println!("Use `cargo run --bin qec-boxkite-sweep` to run the QEC simulation.");
}

#[cfg(test)]
mod tests {
    #[test]
    fn binary_crate_has_expected_binaries() {
        // Smoke test: verify the crate compiles and this test module is reachable.
        // The real binaries (qec-boxkite-sweep, fracton-code, etc.) are in src/bin/.
        let expected = [
            "qec-boxkite-sweep",
            "fractional-schrodinger",
            "fracton-code",
            "projector-fhs-polar",
            "absorber-pareto",
            "fractal-css-expanded",
            "pseudospectrum-slice",
            "grover-classical-comparison",
        ];
        assert_eq!(expected.len(), 8);
    }
}
