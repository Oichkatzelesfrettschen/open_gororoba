//! Exact-decimal verification of the fixed CUJET native-stream extraction.
//! The checker validates arithmetic and duplicated paths, not calibration independence.
use num_bigint::BigInt;
use num_rational::Ratio;
use std::{env, fs, path::Path};
type Rational = Ratio<BigInt>;
fn rational(text: &str) -> Rational {
    let (mantissa, exponent) = text
        .split_once(['e', 'E'])
        .map_or((text, 0), |(number, power)| {
            (number, power.parse::<i32>().unwrap())
        });
    let decimals = mantissa
        .split_once('.')
        .map_or(0, |(_, tail)| tail.len() as i32);
    let numerator: BigInt = mantissa.replace('.', "").parse().unwrap();
    let scale = decimals - exponent;
    let power = BigInt::from(10).pow(scale.unsigned_abs());
    if scale >= 0 {
        Rational::new(numerator, power)
    } else {
        Rational::from_integer(numerator * power)
    }
}
fn absolute(value: Rational) -> Rational {
    if value < rational("0") { -value } else { value }
}
#[derive(Clone, PartialEq)]
struct Curve {
    points: Vec<(Rational, Rational)>,
    radius: Rational,
}
fn curves(source: &str) -> Vec<Curve> {
    let marker = "0.291656 0.291656 0.999985 RG";
    let mut curves = Vec::new();
    for section in source.split(marker).skip(1) {
        let tokens: Vec<_> = section.split_whitespace().collect();
        let end = tokens.iter().position(|token| *token == "S").unwrap();
        let tokens = &tokens[..end];
        let Some(miter_position) = tokens.iter().position(|token| *token == "M") else {
            continue;
        };
        let width_position = tokens.iter().position(|token| *token == "w").unwrap();
        let radius = rational(tokens[width_position - 1]) * rational(tokens[miter_position - 1])
            / rational("2");
        let path = &tokens[miter_position + 1..];
        assert_eq!(path.len() % 3, 0, "fixed-source path grammar");
        let points: Vec<_> = path
            .chunks_exact(3)
            .enumerate()
            .map(|(index, point)| {
                assert_eq!(point[2], if index == 0 { "m" } else { "l" });
                (rational(point[0]), rational(point[1]))
            })
            .collect();
        if points.len() <= 20 {
            continue;
        }
        assert!(points.windows(2).all(|pair| pair[0].0 < pair[1].0));
        curves.push(Curve { points, radius });
    }
    assert_eq!(curves.len(), 6);
    for index in 0..3 {
        assert!(
            curves[index] == curves[index + 3],
            "duplicate path and stroke identity"
        );
    }
    curves.truncate(3);
    curves
}
fn native_x(momentum: &Rational) -> Rational {
    rational("177.461") + (momentum - rational("20")) * rational("580.039") / rational("100")
}
fn momentum(native: &Rational) -> Rational {
    rational("20") + (native - rational("177.461")) * rational("100") / rational("580.039")
}
fn raa(native: &Rational) -> Rational {
    (native - rational("82.285")) / rational("423.305")
}
fn interpolate(curve: &Curve, position: &Rational) -> Rational {
    let pair = curve
        .points
        .windows(2)
        .find(|pair| pair[0].0 <= *position && *position <= pair[1].0)
        .expect("represented path domain");
    let (left, right) = (&pair[0], &pair[1]);
    &left.1 + (&right.1 - &left.1) * (position - &left.0) / (&right.0 - &left.0)
}
fn envelope(
    curve: &Curve,
    low: &Rational,
    high: &Rational,
    graphical: bool,
) -> (Rational, Rational) {
    let radius = if graphical {
        curve.radius.clone()
    } else {
        rational("0")
    };
    let left = native_x(low) - &radius;
    let right = native_x(high) + &radius;
    let mut ordinates = vec![interpolate(curve, &left), interpolate(curve, &right)];
    ordinates.extend(
        curve
            .points
            .iter()
            .filter(|point| left <= point.0 && point.0 <= right)
            .map(|point| point.1.clone()),
    );
    (
        raa(&(ordinates.iter().min().unwrap() - &radius)),
        raa(&(ordinates.iter().max().unwrap() + &radius)),
    )
}
struct Check {
    count: usize,
    maximum_error: Rational,
}
impl Check {
    fn value(&mut self, printed: &str, exact: Rational) {
        let error = absolute(rational(printed) - exact);
        assert!(
            error < rational("1e-12"),
            "reported value exceeds exact arithmetic bound: {printed}"
        );
        self.maximum_error = self.maximum_error.clone().max(error);
        self.count += 1;
    }
}
fn rows(path: &Path) -> Vec<Vec<String>> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| line.split(',').map(str::to_string).collect())
        .collect()
}
fn main() {
    let root = env::args().nth(1).expect("evidence directory");
    let root = Path::new(&root);
    let source = fs::read_to_string(root.join("panel-a-native.stream")).unwrap();
    let curves = curves(&source);
    let mut check = Check {
        count: 0,
        maximum_error: rational("0"),
    };
    let vertices = rows(&root.join("extraction/source-vertices.csv"));
    assert_eq!(
        vertices.len(),
        curves.iter().map(|curve| curve.points.len()).sum::<usize>()
    );
    let mut cursor = 0;
    for (scheme, curve) in curves.iter().enumerate() {
        for (index, point) in curve.points.iter().enumerate() {
            let row = &vertices[cursor];
            cursor += 1;
            assert_eq!(row.len(), 7);
            assert_eq!(row[0].parse::<usize>().unwrap(), scheme + 1);
            assert_eq!(row[1].parse::<usize>().unwrap(), index);
            let pt = momentum(&point.0);
            check.value(&row[2], point.0.clone());
            check.value(&row[3], point.1.clone());
            check.value(&row[4], pt.clone());
            check.value(&row[5], raa(&point.1));
            assert_eq!(
                row[6],
                (rational("8") <= pt && pt <= rational("120")).to_string()
            );
        }
    }
    let reference = rows(&root.join("hepdata59944v1-table19.csv"));
    assert_eq!(reference.len(), 65);
    let selected: Vec<_> = reference
        .iter()
        .filter(|row| rational(&row[0]) >= rational("8") && rational(&row[1]) <= rational("120"))
        .collect();
    let comparisons = rows(&root.join("extraction/matched-bin-envelopes.csv"));
    assert_eq!(comparisons.len(), curves.len() * selected.len());
    let mut common = Vec::new();
    for (scheme, curve) in curves.iter().enumerate() {
        let mut common_low = rational("-0.0464");
        let mut common_high = rational("0.0464");
        let mut excluded_point_count = 0;
        for (index, original) in selected.iter().enumerate() {
            let row = &comparisons[scheme * selected.len() + index];
            assert_eq!(row.len(), 15);
            assert_eq!(row[0].parse::<usize>().unwrap(), scheme + 1);
            for field in 0..6 {
                check.value(&row[field + 1], rational(&original[field]));
            }
            let low = rational(&original[0]);
            let high = rational(&original[1]);
            let observed = rational(&original[2]);
            let uncertainty = rational(&original[3]) + rational(&original[4]);
            check.value(
                &row[7],
                raa(&interpolate(
                    curve,
                    &native_x(&((&low + &high) / rational("2"))),
                )),
            );
            let plain = envelope(curve, &low, &high, false);
            let graphic = envelope(curve, &low, &high, true);
            check.value(&row[8], plain.0);
            check.value(&row[9], plain.1);
            check.value(&row[10], graphic.0.clone());
            check.value(&row[11], graphic.1.clone());
            let calibration_overlap = low <= rational("12.5") && rational("12.5") <= high;
            assert_eq!(row[12], calibration_overlap.to_string());
            let feasible_low = (graphic.0 - &uncertainty) / &observed - rational("1");
            let feasible_high = (graphic.1 + &uncertainty) / &observed - rational("1");
            check.value(&row[13], feasible_low.clone());
            check.value(&row[14], feasible_high.clone());
            if !calibration_overlap {
                excluded_point_count += 1;
                common_low = common_low.max(feasible_low);
                common_high = common_high.min(feasible_high);
            }
        }
        assert_eq!(excluded_point_count, 20);
        common.push((common_low, common_high));
    }
    let summary = fs::read_to_string(root.join("extraction/summary.toml")).unwrap();
    let sections: Vec<_> = summary.split("[[scheme]]").skip(1).collect();
    assert_eq!(sections.len(), 3);
    for (section, (low, high)) in sections.iter().zip(&common) {
        let field = |name: &str| {
            section
                .lines()
                .find_map(|line| {
                    line.split_once(" = ")
                        .filter(|(key, _)| *key == name)
                        .map(|(_, value)| value)
                })
                .expect("summary field")
        };
        check.value(field("common_normalization_low"), low.clone());
        check.value(field("common_normalization_high"), high.clone());
        assert_eq!(field("component_box_compatible"), (low <= high).to_string());
    }
    println!("checked_values = {}", check.count);
    println!("strict_absolute_error_bound = \"1e-12\"");
    println!(
        "maximum_error_numerator = \"{}\"",
        check.maximum_error.numer()
    );
    println!(
        "maximum_error_denominator = \"{}\"",
        check.maximum_error.denom()
    );
    println!("all_exact_errors_below_bound = true");
    println!(
        "scope = \"Exact decimal arithmetic and full retained row inventory; shared source axes and observational calibration remain separate evidence\""
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_decimal_and_interpolation_controls() {
        assert_eq!(rational("1.25e-2"), rational("0.0125"));
        assert_eq!(rational("-2.5E2"), rational("-250"));
        let curve = Curve { points: vec![(rational("0"), rational("1")), (rational("3"), rational("2"))], radius: rational("0") };
        assert_eq!(interpolate(&curve, &rational("1")), rational("4") / rational("3"));
    }
    #[test]
    #[should_panic(expected = "reported value exceeds exact arithmetic bound")]
    fn equality_at_error_bound_fails() {
        Check { count: 0, maximum_error: rational("0") }.value("1e-12", rational("0"));
    }
    #[test]
    #[should_panic(expected = "duplicate path and stroke identity")]
    fn changed_duplicate_is_rejected() {
        let changed = include_str!("panel-a-native.stream").replacen("102.055 171.059 m", "102.055 171.060 m", 1);
        curves(&changed);
    }
}
