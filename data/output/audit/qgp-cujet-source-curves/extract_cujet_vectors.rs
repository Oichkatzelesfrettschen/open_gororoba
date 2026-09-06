use std::{env, fs, io::Write, path::Path};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(Clone, Debug)]
struct State {
    color: [f64; 3],
    width: f64,
    dash: Vec<f64>,
    miter: f64,
}

#[derive(Clone, Debug)]
struct Curve {
    points: Vec<Point>,
    state: State,
}

#[derive(Debug)]
enum Token {
    Number(f64),
    Array(String),
    Opaque,
    Operator(String),
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    condition.then_some(()).ok_or_else(|| message.to_string())
}

fn tokenize(source: &str) -> Result<Vec<Token>, String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        let start = position;
        match bytes[position] {
            byte if byte.is_ascii_whitespace() => position += 1,
            b'%' => {
                while position < bytes.len() && bytes[position] != b'\n' {
                    position += 1;
                }
            }
            b'(' => {
                let mut depth = 1;
                position += 1;
                while position < bytes.len() && depth > 0 {
                    match bytes[position] {
                        b'\\' => position += 1,
                        b'(' => depth += 1,
                        b')' => depth -= 1,
                        _ => {}
                    }
                    position += 1;
                }
                require(
                    depth == 0 && position <= bytes.len(),
                    "unterminated PDF string",
                )?;
                tokens.push(Token::Opaque);
            }
            b'[' | b'<' => {
                let terminator = if bytes[position] == b'[' { b']' } else { b'>' };
                position += 1;
                while position < bytes.len() && bytes[position] != terminator {
                    position += 1;
                }
                require(
                    position < bytes.len(),
                    "unterminated PDF array or hex string",
                )?;
                if terminator == b']' {
                    tokens.push(Token::Array(source[start + 1..position].to_string()));
                } else {
                    tokens.push(Token::Opaque);
                }
                position += 1;
            }
            _ => {
                position += 1;
                while position < bytes.len()
                    && !bytes[position].is_ascii_whitespace()
                    && !b"()[]<>/%".contains(&bytes[position])
                {
                    position += 1;
                }
                let word = &source[start..position];
                if word.starts_with('/') {
                    tokens.push(Token::Opaque);
                } else if let Ok(number) = word.parse::<f64>() {
                    require(number.is_finite(), "nonfinite PDF operand")?;
                    tokens.push(Token::Number(number));
                } else {
                    tokens.push(Token::Operator(word.to_string()));
                }
            }
        }
    }
    Ok(tokens)
}

fn numbers(operands: &[Token], count: usize) -> Result<Vec<f64>, String> {
    require(operands.len() == count, "PDF operand count")?;
    operands
        .iter()
        .map(|token| match token {
            Token::Number(number) => Ok(*number),
            _ => Err("expected numeric PDF operand".to_string()),
        })
        .collect()
}

fn extract(source: &str) -> Result<Vec<Curve>, String> {
    let mut state = State {
        color: [0.0; 3],
        width: 1.0,
        dash: Vec::new(),
        miter: 10.0,
    };
    let mut stack = Vec::new();
    let mut operands = Vec::new();
    let mut points = Vec::new();
    let mut nonlinear = false;
    let mut curves = Vec::new();
    let mut axis_segments = Vec::new();
    for token in tokenize(source)? {
        let Token::Operator(operator) = token else {
            operands.push(token);
            continue;
        };
        match operator.as_str() {
            "q" => stack.push(state.clone()),
            "Q" => state = stack.pop().ok_or("unbalanced graphics state")?,
            "cm" => return Err("native panel transform requires separate admission".into()),
            "RG" => {
                let values = numbers(&operands, 3)?;
                state.color.copy_from_slice(&values);
            }
            "G" => state.color = [numbers(&operands, 1)?[0]; 3],
            "K" => state.color = [-1.0; 3],
            "w" => state.width = numbers(&operands, 1)?[0],
            "M" => state.miter = numbers(&operands, 1)?[0],
            "d" => {
                require(operands.len() == 2, "dash operands")?;
                let Token::Array(array) = &operands[0] else {
                    return Err("dash array missing".into());
                };
                state.dash = array
                    .split_whitespace()
                    .map(|value| value.parse::<f64>().map_err(|_| "invalid dash".into()))
                    .collect::<Result<_, String>>()?;
            }
            "m" => {
                let values = numbers(&operands, 2)?;
                points.clear();
                nonlinear = false;
                points.push(Point {
                    x: values[0],
                    y: values[1],
                });
            }
            "l" => {
                let values = numbers(&operands, 2)?;
                points.push(Point {
                    x: values[0],
                    y: values[1],
                });
            }
            "c" | "v" | "y" | "h" | "re" => nonlinear = true,
            "S" => {
                if state.color == [0.291656, 0.291656, 0.999985] && points.len() > 20 {
                    require(
                        !nonlinear,
                        "source curve contains unsupported path geometry",
                    )?;
                    require(
                        points.windows(2).all(|pair| pair[0].x < pair[1].x),
                        "curve x order",
                    )?;
                    require(
                        state.width > 0.0 && state.miter >= 1.0,
                        "stroke envelope parameters",
                    )?;
                    curves.push(Curve {
                        points: points.clone(),
                        state: state.clone(),
                    });
                }
                if state.color == [0.399994; 3] && points.len() == 2 && !nonlinear {
                    axis_segments.push((points[0], points[1]));
                }
                points.clear();
                nonlinear = false;
            }
            "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n" => {
                points.clear();
                nonlinear = false;
            }
            _ => {}
        }
        operands.clear();
    }
    require(stack.is_empty(), "unclosed graphics state")?;
    for x in [177.461, 293.469, 409.477, 525.484, 641.492, 757.5] {
        require(
            axis_segments.contains(&(Point { x, y: 73.18 }, Point { x, y: 77.172 })),
            "x tick identity",
        )?;
    }
    for y in [82.285, 166.945, 251.605, 336.27, 420.93, 505.59] {
        require(
            axis_segments.contains(&(Point { x: 76.801, y }, Point { x: 80.793, y })),
            "y tick identity",
        )?;
    }
    require(
        curves.len() == 6,
        "expected two copies of each of three blue curves",
    )?;
    for index in 0..3 {
        require(
            curves[index].points == curves[index + 3].points,
            "duplicate curve mismatch",
        )?;
        require(
            curves[index].state.dash == curves[index + 3].state.dash,
            "duplicate style mismatch",
        )?;
    }
    require(curves[0].state.dash.is_empty(), "solid scheme identity")?;
    require(curves[1].state.dash == [5.2, 5.2], "dashed scheme identity")?;
    require(
        curves[2].state.dash == [1.3, 5.2, 5.2, 5.2],
        "dot-dashed scheme identity",
    )?;
    curves.truncate(3);
    Ok(curves)
}

fn native_x(pt: f64) -> f64 {
    177.461 + (pt - 20.0) * (757.5 - 177.461) / 100.0
}
fn pt(x: f64) -> f64 {
    20.0 + (x - 177.461) * 100.0 / (757.5 - 177.461)
}
fn raa(y: f64) -> f64 {
    (y - 82.285) / (505.59 - 82.285)
}
fn interpolate(points: &[Point], x: f64) -> Result<f64, String> {
    let pair = points
        .windows(2)
        .find(|pair| pair[0].x <= x && x <= pair[1].x)
        .ok_or("query outside represented polyline")?;
    Ok(pair[0].y + (pair[1].y - pair[0].y) * (x - pair[0].x) / (pair[1].x - pair[0].x))
}
fn range(curve: &Curve, low: f64, high: f64, graphical: bool) -> Result<(f64, f64), String> {
    // The miter limit bounds stroke excursions in native drawing coordinates.
    // The envelope also spans the whole bin, independently of a pp weighting.
    let radius = if graphical {
        curve.state.width * curve.state.miter / 2.0
    } else {
        0.0
    };
    let left = native_x(low) - radius;
    let right = native_x(high) + radius;
    let mut values = vec![
        interpolate(&curve.points, left)?,
        interpolate(&curve.points, right)?,
    ];
    values.extend(
        curve
            .points
            .iter()
            .filter(|p| left <= p.x && p.x <= right)
            .map(|p| p.y),
    );
    Ok((
        raa(values.iter().copied().fold(f64::INFINITY, f64::min) - radius),
        raa(values.iter().copied().fold(f64::NEG_INFINITY, f64::max) + radius),
    ))
}

fn new_file(path: &Path) -> Result<fs::File, std::io::Error> {
    fs::File::create_new(path)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = env::args().nth(1).ok_or("expected evidence directory")?;
    let output = env::args()
        .nth(2)
        .ok_or("expected fresh output directory")?;
    let directory = Path::new(&directory);
    let output = Path::new(&output);
    let source = fs::read_to_string(directory.join("panel-a-native.stream"))?;
    let curves = extract(&source)?;
    let reference = fs::read_to_string(directory.join("hepdata59944v1-table19.csv"))?;
    require(
        reference.lines().next()
            == Some("pt_low_gev,pt_high_gev,raa,stat,sys,normalization_fraction"),
        "reference schema",
    )?;
    let mut rows = Vec::new();
    for line in reference.lines().skip(1) {
        let values = line
            .split(',')
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        require(
            values.len() == 6 && values.iter().all(|x| x.is_finite()),
            "reference values",
        )?;
        require(
            values[0] < values[1]
                && values[2] > 0.0
                && values[3] >= 0.0
                && values[4] >= 0.0
                && values[5] == 0.0464,
            "reference admission",
        )?;
        rows.push(values);
    }
    require(
        rows.windows(2).all(|pair| pair[0][1] <= pair[1][0]),
        "reference bin ordering",
    )?;
    fs::create_dir(output)?;
    let mut vertices = new_file(&output.join("source-vertices.csv"))?;
    writeln!(
        vertices,
        "scheme,vertex,native_x,native_y,pt_gev,raa,source_domain"
    )?;
    for (scheme, curve) in curves.iter().enumerate() {
        for (vertex, point) in curve.points.iter().enumerate() {
            let momentum = pt(point.x);
            writeln!(
                vertices,
                "{},{vertex},{:.17},{:.17},{momentum:.17},{:.17},{}",
                scheme + 1,
                point.x,
                point.y,
                raa(point.y),
                (8.0..=120.0).contains(&momentum)
            )?;
        }
    }
    let mut comparisons = new_file(&output.join("matched-bin-envelopes.csv"))?;
    writeln!(
        comparisons,
        "scheme,pt_low_gev,pt_high_gev,observed_raa,stat,sys,normalization_fraction,centerline_midpoint,centerline_bin_low,centerline_bin_high,graphic_bin_low,graphic_bin_high,calibration_overlap,normalization_feasible_low,normalization_feasible_high"
    )?;
    let mut summary = new_file(&output.join("summary.toml"))?;
    writeln!(
        summary,
        "reference_rows = {}\nsource_curve_vertices = [{}, {}, {}]\nsource_curve_copies = 2\nlocal_fitted_parameters = 0\nstatistical_model_ranking = \"unadmitted\"",
        rows.len(),
        curves[0].points.len(),
        curves[1].points.len(),
        curves[2].points.len()
    )?;
    for (scheme, curve) in curves.iter().enumerate() {
        let mut common_low = -0.0464_f64;
        let mut common_high = 0.0464_f64;
        let mut calibration_point_excluded_count = 0;
        let mut selected_count = 0;
        for row in &rows {
            let [low, high, observed, stat, sys, normalization] = row.as_slice() else {
                unreachable!()
            };
            if *low < 8.0 || *high > 120.0 {
                continue;
            }
            selected_count += 1;
            let midpoint = raa(interpolate(&curve.points, native_x((low + high) / 2.0))?);
            let (center_low, center_high) = range(curve, *low, *high, false)?;
            let (graphic_low, graphic_high) = range(curve, *low, *high, true)?;
            let calibration_overlap = *low <= 12.5 && 12.5 <= *high;
            let normalization_low = (graphic_low - stat - sys) / observed - 1.0;
            let normalization_high = (graphic_high + stat + sys) / observed - 1.0;
            if !calibration_overlap {
                calibration_point_excluded_count += 1;
                common_low = common_low.max(normalization_low);
                common_high = common_high.min(normalization_high);
            }
            writeln!(
                comparisons,
                "{},{low},{high},{observed},{stat},{sys},{normalization},{midpoint:.17},{center_low:.17},{center_high:.17},{graphic_low:.17},{graphic_high:.17},{calibration_overlap},{normalization_low:.17},{normalization_high:.17}",
                scheme + 1
            )?;
        }
        writeln!(
            summary,
            "\n[[scheme]]\nid = {}\nselected_bins = {selected_count}\ncalibration_point_excluded_bins = {calibration_point_excluded_count}\ncommon_normalization_low = {common_low:.17}\ncommon_normalization_high = {common_high:.17}\ncomponent_box_compatible = {}",
            scheme + 1,
            common_low <= common_high
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexer_keeps_literal_operators_opaque() {
        let tokens = tokenize("(q RG S (nested) \\) text) [5.2 5.2] 0 d 1 2 m").unwrap();
        assert!(matches!(tokens[0], Token::Opaque));
        assert!(matches!(tokens[1], Token::Array(_)));
        assert!(tokenize("(unfinished").is_err());
    }

    #[test]
    fn source_mutations_fail_admission() {
        let source = include_str!("panel-a-native.stream");
        assert_eq!(extract(source).unwrap().len(), 3);
        for changed in [
            source.replace("0.291656 0.291656 0.999985 RG", "0.291656 0.291656 0.5 RG"),
            source.replace("177.461 73.18 m", "178.461 73.18 m"),
            source.replacen("102.055 171.059 m", "103.055 171.059 m", 1),
            source.replace("[ 5.2 5.2]", "[ 6.2 5.2]"),
        ] {
            assert!(extract(&changed).is_err());
        }
    }

    #[test]
    fn interpolation_and_bin_envelope_have_independent_controls() {
        let curve = Curve {
            points: vec![
                Point { x: 0.0, y: 0.0 },
                Point {
                    x: 1000.0,
                    y: 500.0,
                },
            ],
            state: State {
                color: [0.0; 3],
                width: 2.0,
                dash: vec![],
                miter: 3.0,
            },
        };
        assert_eq!(interpolate(&curve.points, 200.0).unwrap(), 100.0);
        assert!(interpolate(&curve.points, -1.0).is_err());
        let plain = range(&curve, 20.0, 40.0, false).unwrap();
        let graphic = range(&curve, 20.0, 40.0, true).unwrap();
        assert!((plain.0 - raa(177.461 / 2.0)).abs() < 1e-14);
        assert!(graphic.0 < plain.0 && graphic.1 > plain.1);
    }
}
