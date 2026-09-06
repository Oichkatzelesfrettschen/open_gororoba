//! Immutable interpolation of externally supplied DSS-format fragmentation grids.
//! The caller declares table provenance; parsing validates shape and numbers,
//! without authenticating an author release or a physical population.

const Z_NODES: [f64; 35] = [
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095, 0.1, 0.125, 0.15, 0.175, 0.2,
    0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.93, 1.0,
];
const Q2_NODES: [f64; 24] = [
    1.0, 1.25, 1.5, 2.5, 4.0, 6.4, 10.0, 15.0, 25.0, 40.0, 64.0, 100.0, 180.0, 320.0, 580.0,
    1000.0, 1800.0, 3200.0, 5800.0, 10000.0, 18000.0, 32000.0, 58000.0, 100000.0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DssHadron {
    Pion,
    Kaon,
    Proton,
    ChargedHadron,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerturbativeOrder {
    Lo,
    Nlo,
}

/// Interpretation of unsuffixed z-node and endpoint-exponent source literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceRealPrecision {
    /// Round z nodes and the 0.3 exponent through binary32 before conversion to binary64.
    DefaultReal32,
    /// Source amendment with default-real64 literals, including the 0.3 exponent.
    DefaultReal64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadronCharge {
    Average,
    Plus,
    Minus,
    Sum,
}

/// Caller-declared labels; a provenance locator is not a verified content hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DssGridIdentity {
    pub hadron: DssHadron,
    pub order: PerturbativeOrder,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentationError(pub &'static str);

impl std::fmt::Display for FragmentationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}
impl std::error::Error for FragmentationError {}

/// Every component is z times D for the selected hadron charge.
/// Charm and bottom each denote one parton; their antipartons have equal values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FragmentationValues {
    pub u: f64,
    pub ubar: f64,
    pub d: f64,
    pub dbar: f64,
    pub s: f64,
    pub sbar: f64,
    pub charm: f64,
    pub bottom: f64,
    pub gluon: f64,
}

pub struct DssGrid {
    identity: DssGridIdentity,
    precision: SourceRealPrecision,
    log_z: [f64; 35],
    log_q2: [f64; 24],
    reduced: Vec<[f64; 9]>,
}

fn endpoint_factor(z: f64, column: usize, precision: SourceRealPrecision) -> f64 {
    let fractional = match precision {
        SourceRealPrecision::DefaultReal32 => f64::from(0.3_f32),
        SourceRealPrecision::DefaultReal64 => 0.3,
    };
    let (power, small_z) = match column {
        3 | 4 => (7, fractional),
        5 => (4, fractional),
        _ => (4, 0.5),
    };
    (1.0 - z).powi(power) * z.powf(small_z)
}

fn interval(nodes: &[f64], value: f64) -> (usize, f64) {
    let upper = nodes
        .partition_point(|node| *node < value)
        .clamp(1, nodes.len() - 1);
    let lower = upper - 1;
    (
        lower,
        (value - nodes[lower]) / (nodes[upper] - nodes[lower]),
    )
}

impl DssGrid {
    /// Read 816 fixed-width records: 34 z nodes, 24 Q2 nodes, nine width-10 fields.
    /// Signed values are retained, including negative NLO components.
    pub fn parse(
        source: &str,
        identity: DssGridIdentity,
        precision: SourceRealPrecision,
    ) -> Result<Self, FragmentationError> {
        if identity.provenance.trim().is_empty() {
            return Err(FragmentationError("grid provenance locator is empty"));
        }
        let z_nodes = Z_NODES.map(|value| match precision {
            SourceRealPrecision::DefaultReal32 => f64::from(value as f32),
            SourceRealPrecision::DefaultReal64 => value,
        });
        let lines: Vec<&str> = source.lines().collect();
        if lines.len() != 34 * 24 {
            return Err(FragmentationError("grid requires exactly816 records"));
        }
        let mut reduced = vec![[0.0; 9]; 35 * 24];
        for (row, line) in lines.into_iter().enumerate() {
            if !line.is_ascii() || line.len() != 90 {
                return Err(FragmentationError(
                    "grid record requires nine ASCII width10 fields",
                ));
            }
            for column in 0..9 {
                let token = line[column * 10..(column + 1) * 10].trim();
                if token.is_empty()
                    || !line[column * 10..(column + 1) * 10].bytes().all(|byte| {
                        byte.is_ascii_digit()
                            || matches!(byte, b' ' | b'+' | b'-' | b'.' | b'e' | b'E')
                    })
                {
                    return Err(FragmentationError(
                        "grid field contains invalid decimal characters",
                    ));
                }
                let value: f64 = token
                    .parse()
                    .map_err(|_| FragmentationError("invalid grid decimal field"))?;
                if !value.is_finite() {
                    return Err(FragmentationError("grid field must be finite"));
                }
                if value == 0.0
                    && token
                        .split(['e', 'E'])
                        .next()
                        .unwrap_or("")
                        .bytes()
                        .any(|byte| (b'1'..=b'9').contains(&byte))
                {
                    return Err(FragmentationError("nonzero grid field underflows binary64"));
                }
                let normalized = value / endpoint_factor(z_nodes[row / 24], column, precision);
                if !normalized.is_finite() {
                    return Err(FragmentationError("grid endpoint normalization overflow"));
                }
                reduced[row][column] = normalized;
            }
        }
        Ok(Self {
            identity,
            precision,
            log_z: z_nodes.map(f64::ln),
            log_q2: Q2_NODES.map(f64::ln),
            reduced,
        })
    }

    pub fn identity(&self) -> &DssGridIdentity {
        &self.identity
    }
    pub fn source_real_precision(&self) -> SourceRealPrecision {
        self.precision
    }

    /// Evaluate the declared zD measure within z=.05..1 and Q2=1..1e5 GeV squared.
    /// Out-of-range and nonfinite inputs are rejected rather than clamped or extrapolated.
    pub fn evaluate(
        &self,
        z: f64,
        q2: f64,
        charge: HadronCharge,
    ) -> Result<FragmentationValues, FragmentationError> {
        if !z.is_finite()
            || !(0.05..=1.0).contains(&z)
            || !q2.is_finite()
            || !(1.0..=1e5).contains(&q2)
        {
            return Err(FragmentationError(
                "fragmentation query outside admitted z/Q2 range",
            ));
        }
        let (z_index, z_fraction) = interval(&self.log_z, z.ln());
        let (q_index, q_fraction) = interval(&self.log_q2, q2.ln());
        let mut columns = [0.0; 9];
        for (column, output) in columns.iter_mut().enumerate() {
            let low = self.reduced[z_index * 24 + q_index][column] * (1.0 - q_fraction)
                + self.reduced[z_index * 24 + q_index + 1][column] * q_fraction;
            let high = self.reduced[(z_index + 1) * 24 + q_index][column] * (1.0 - q_fraction)
                + self.reduced[(z_index + 1) * 24 + q_index + 1][column] * q_fraction;
            *output = (low * (1.0 - z_fraction) + high * z_fraction)
                * endpoint_factor(z, column, self.precision);
        }
        let (sign, factor) = match charge {
            HadronCharge::Plus => (1.0, 0.5),
            HadronCharge::Minus => (-1.0, 0.5),
            HadronCharge::Average => (0.0, 0.5),
            HadronCharge::Sum => (0.0, 1.0),
        };
        let values = FragmentationValues {
            u: factor * (columns[0] + sign * columns[6]),
            ubar: factor * (columns[0] - sign * columns[6]),
            d: factor * (columns[1] + sign * columns[7]),
            dbar: factor * (columns[1] - sign * columns[7]),
            s: factor * (columns[2] + sign * columns[8]),
            sbar: factor * (columns[2] - sign * columns[8]),
            charm: factor * columns[3],
            bottom: factor * columns[4],
            gluon: 2.0 * factor * columns[5],
        };
        if [
            values.u,
            values.ubar,
            values.d,
            values.dbar,
            values.s,
            values.sbar,
            values.charm,
            values.bottom,
            values.gluon,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(FragmentationError(
                "fragmentation interpolation or charge reconstruction overflow",
            ));
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> String {
        let row = [
            " 2.000E+00",
            " 4.000E+00",
            " 6.000E+00",
            " 8.000E+00",
            " 1.000E+01",
            " 3.000E+00",
            " 1.000E+00",
            "-2.000E+00",
            " 0.000E+00",
        ]
        .join("");
        (0..816).map(|_| format!("{row}\n")).collect()
    }
    fn identity() -> DssGridIdentity {
        DssGridIdentity {
            hadron: DssHadron::ChargedHadron,
            order: PerturbativeOrder::Nlo,
            provenance: "synthetic signed fixed-width fixture".into(),
        }
    }

    #[test]
    fn nodes_resolve_charge_and_measure() {
        let grid =
            DssGrid::parse(&fixture(), identity(), SourceRealPrecision::DefaultReal64).unwrap();
        let plus = grid.evaluate(0.5, 100.0, HadronCharge::Plus).unwrap();
        let minus = grid.evaluate(0.5, 100.0, HadronCharge::Minus).unwrap();
        let sum = grid.evaluate(0.5, 100.0, HadronCharge::Sum).unwrap();
        for (actual, expected) in [plus.u, plus.ubar, plus.charm, plus.gluon]
            .into_iter()
            .zip([1.5, 0.5, 4.0, 3.0])
        {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert_eq!(minus.u, plus.ubar);
        assert_eq!(sum.u, plus.u + minus.u);
        assert_eq!(sum.gluon, 6.0);
        assert_eq!(grid.evaluate(1.0, 100.0, HadronCharge::Sum).unwrap().u, 0.0);
    }

    #[test]
    fn rejects_bad_shape_values_and_queries() {
        let text = fixture();
        assert!(
            DssGrid::parse(
                &text[..text.len() - 91],
                identity(),
                SourceRealPrecision::DefaultReal64
            )
            .is_err()
        );
        let invalid = text.replacen(" 2.000E+00", "       NaN", 1);
        assert!(DssGrid::parse(&invalid, identity(), SourceRealPrecision::DefaultReal64).is_err());
        let grid = DssGrid::parse(&text, identity(), SourceRealPrecision::DefaultReal32).unwrap();
        for (z, q2) in [
            (f64::NAN, 1.0),
            (0.04, 1.0),
            (1.01, 1.0),
            (0.5, 0.0),
            (0.5, f64::INFINITY),
        ] {
            assert!(grid.evaluate(z, q2, HadronCharge::Average).is_err());
        }
    }
}
