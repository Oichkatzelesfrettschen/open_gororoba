//! Exact lexical admission of one dependent group in HEPData table JSON.
//!
//! Population qualifiers and their group spans identify the selected observable.
//! Numeric parsing validates admission while original strings carry the output.
//! The parser accepts one binned independent variable and symmetric errors.
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;
use std::{cmp::Ordering, collections::BTreeSet, fmt};

struct UniqueValue(Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Bool(value)))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Number(value.into())))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Number(value.into())))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueValue(Value::Number(number)))
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value.to_owned())))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::String(value)))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueValue(value)) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(UniqueValue(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut mapping: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = mapping.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!(
                            "duplicate JSON object key {key}"
                        )));
                    }
                    let UniqueValue(value) = mapping.next_value()?;
                    values.insert(key, value);
                }
                Ok(UniqueValue(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

pub struct HepDataTableContract<'a> {
    pub doi: &'a str,
    pub independent_header: &'a str,
    pub dependent_header: &'a str,
    pub group: usize,
    pub qualifiers: &'a [(&'a str, &'a str)],
    pub required_errors: &'a [&'a str],
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HepDataLexicalError {
    pub label: String,
    pub symerror: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HepDataLexicalRow {
    pub low: String,
    pub high: String,
    pub value: String,
    pub errors: Vec<HepDataLexicalError>,
}
fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn integer(value: &Value, field: &str) -> Result<usize, String> {
    value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("invalid {field}"))
}
fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("missing string {field}"))
}
fn decimal(value: &Value, field: &str) -> Result<(String, f64), String> {
    let lexical = text(value, field)?;
    // Decimal notation admits sign, decimal point and exponent but excludes
    // whitespace, special float names and nondecimal input conventions.
    require(
        !lexical.is_empty()
            && lexical
                .bytes()
                .all(|byte| byte.is_ascii_digit() || b"+-.eE".contains(&byte)),
        "invalid decimal spelling",
    )?;
    let number = lexical
        .parse::<f64>()
        .map_err(|_| format!("invalid decimal {field}"))?;
    require(number.is_finite(), "nonfinite decimal")?;
    let nonzero_mantissa = lexical
        .split(['e', 'E'])
        .next()
        .is_some_and(|mantissa| mantissa.bytes().any(|byte| (b'1'..=b'9').contains(&byte)));
    require(
        number != 0.0 || !nonzero_mantissa,
        "unrepresentable nonzero decimal",
    )?;
    Ok((lexical.to_owned(), number))
}

// Significant decimal digits and their leading base-ten position suffice for
// exact ordering; padding with zeroes avoids constructing large integer powers.
fn normalized_decimal(value: &str) -> Result<(bool, i64, Vec<u8>), String> {
    let (mantissa, exponent) = match value.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (
            mantissa,
            exponent
                .parse::<i64>()
                .map_err(|_| "decimal exponent outside ordering range")?,
        ),
        None => (value, 0),
    };
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.trim_start_matches(['-', '+']);
    let fraction_length = unsigned
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let all_digits: Vec<_> = unsigned.bytes().filter(|byte| *byte != b'.').collect();
    let Some(first) = all_digits.iter().position(|byte| *byte != b'0') else {
        return Ok((false, 0, Vec::new()));
    };
    let digits = all_digits[first..].to_vec();
    let position = exponent
        .checked_sub(i64::try_from(fraction_length).map_err(|_| "decimal fraction too long")?)
        .and_then(|position| position.checked_add(i64::try_from(digits.len()).ok()?))
        .ok_or("decimal ordering exponent overflow")?;
    Ok((negative, position, digits))
}

fn decimal_order(left: &str, right: &str) -> Result<Ordering, String> {
    let (left_negative, left_position, left_digits) = normalized_decimal(left)?;
    let (right_negative, right_position, right_digits) = normalized_decimal(right)?;
    if left_negative != right_negative {
        return Ok(if left_negative {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }
    let magnitude = match (left_digits.is_empty(), right_digits.is_empty()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => left_position.cmp(&right_position).then_with(|| {
            (0..left_digits.len().max(right_digits.len()))
                .map(|index| {
                    left_digits
                        .get(index)
                        .unwrap_or(&b'0')
                        .cmp(right_digits.get(index).unwrap_or(&b'0'))
                })
                .find(|ordering| *ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        }),
    };
    Ok(if left_negative {
        magnitude.reverse()
    } else {
        magnitude
    })
}

/// Admit one explicitly identified group without fetching or writing artifacts.
pub fn admit_table_json(
    source: &str,
    contract: &HepDataTableContract<'_>,
) -> Result<Vec<HepDataLexicalRow>, String> {
    require(
        !contract.doi.is_empty()
            && !contract.independent_header.is_empty()
            && !contract.dependent_header.is_empty(),
        "empty source identity",
    )?;
    let required_errors: BTreeSet<_> = contract.required_errors.iter().copied().collect();
    require(
        !required_errors.is_empty()
            && required_errors.len() == contract.required_errors.len()
            && required_errors.iter().all(|label| !label.is_empty()),
        "invalid required-error contract",
    )?;
    let qualifier_names: BTreeSet<_> = contract.qualifiers.iter().map(|(name, _)| *name).collect();
    require(
        !qualifier_names.is_empty()
            && qualifier_names.len() == contract.qualifiers.len()
            && contract
                .qualifiers
                .iter()
                .all(|(name, value)| !name.is_empty() && !value.is_empty()),
        "invalid qualifier contract",
    )?;
    let UniqueValue(document) =
        serde_json::from_str(source).map_err(|error| format!("table JSON: {error}"))?;
    require(
        text(&document["doi"], "DOI")? == contract.doi,
        "DOI mismatch",
    )?;
    let headers = document["headers"].as_array().ok_or("missing headers")?;
    require(
        headers.len() == 2 && integer(&headers[0]["colspan"], "independent header width")? == 1,
        "one independent and one dependent header required",
    )?;
    require(
        text(&headers[0]["name"], "independent header")? == contract.independent_header,
        "independent-header mismatch",
    )?;
    require(
        text(&headers[1]["name"], "dependent header")? == contract.dependent_header,
        "dependent-header mismatch",
    )?;
    let groups = integer(&headers[1]["colspan"], "dependent header width")?;
    require(
        groups > 0 && contract.group < groups,
        "selected group outside header",
    )?;
    let qualifiers = document["qualifiers"]
        .as_object()
        .ok_or("missing qualifiers")?;
    for (name, expected) in contract.qualifiers {
        let entries = qualifiers
            .get(*name)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing qualifier {name}"))?;
        let mut spans = Vec::new();
        let mut selected = None;
        for entry in entries {
            require(
                text(&entry["type"], "qualifier type")? == *name,
                "qualifier type mismatch",
            )?;
            let start = integer(&entry["group"], "qualifier group")?;
            let width = integer(&entry["colspan"], "qualifier span")?;
            let end = start.checked_add(width).ok_or("qualifier span overflow")?;
            require(width > 0 && end <= groups, "qualifier span outside header")?;
            require(
                spans
                    .iter()
                    .all(|&(other_start, other_end)| end <= other_start || other_end <= start),
                "overlapping qualifier spans",
            )?;
            spans.push((start, end));
            let value = text(&entry["value"], "qualifier value")?;
            if start <= contract.group && contract.group < end {
                selected = Some(value);
            }
        }
        require(
            selected == Some(*expected),
            "selected population qualifier mismatch or missing coverage",
        )?;
    }
    let rows = document["values"].as_array().ok_or("missing values")?;
    require(!rows.is_empty(), "empty table")?;
    let mut admitted = Vec::with_capacity(rows.len());
    let mut previous_high: Option<String> = None;
    for row in rows {
        let coordinates = row["x"]
            .as_array()
            .ok_or("missing independent coordinates")?;
        require(
            coordinates.len() == 1,
            "one independent coordinate required",
        )?;
        let (low, low_number) = decimal(&coordinates[0]["low"], "bin low")?;
        let (high, high_number) = decimal(&coordinates[0]["high"], "bin high")?;
        require(
            low_number < high_number && (high_number - low_number).is_finite(),
            "nonpositive or unrepresentable bin width",
        )?;
        require(
            decimal_order(&low, &high)? == Ordering::Less,
            "nonpositive exact bin width",
        )?;
        if let Some(previous) = &previous_high {
            require(
                decimal_order(previous, &low)? != Ordering::Greater,
                "unordered or overlapping bins",
            )?;
        }
        previous_high = Some(high.clone());
        let values = row["y"].as_array().ok_or("missing dependent values")?;
        let mut seen_groups = BTreeSet::new();
        let mut selected = None;
        for value in values {
            let group = integer(&value["group"], "row group")?;
            require(
                group < groups && seen_groups.insert(group),
                "invalid or duplicate row group",
            )?;
            if group == contract.group {
                selected = Some(value);
            }
        }
        let selected = selected.ok_or("missing selected row group")?;
        let (value, _) = decimal(&selected["value"], "dependent value")?;
        let errors = selected["errors"].as_array().ok_or("missing errors")?;
        let mut seen_labels = BTreeSet::new();
        let mut retained_errors = Vec::with_capacity(errors.len());
        for error in errors {
            let label = text(&error["label"], "error label")?;
            require(
                required_errors.contains(label) && seen_labels.insert(label),
                "unexpected or duplicate selected error",
            )?;
            require(
                error.get("asymerror").is_none(),
                "asymmetric error outside contract",
            )?;
            let (symerror, error_number) = decimal(&error["symerror"], "symmetric error")?;
            require(error_number >= 0.0, "negative error magnitude")?;
            retained_errors.push(HepDataLexicalError {
                label: label.to_owned(),
                symerror,
            });
        }
        require(
            seen_labels == required_errors,
            "missing required selected error",
        )?;
        admitted.push(HepDataLexicalRow {
            low,
            high,
            value,
            errors: retained_errors,
        });
    }
    Ok(admitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn fixture() -> Value {
        json!({"doi":"source/t4", "headers":[{"name":"PT","colspan":1},{"name":"yield","colspan":2}],
        "qualifiers":{"RE":[{"type":"RE","group":0,"colspan":2,"value":"PP"}],"ENERGY":[{"type":"ENERGY","group":0,"colspan":1,"value":"5020"},{"type":"ENERGY","group":1,"colspan":1,"value":"2760"}]},
        "values":[{"x":[{"low":"0.15","high":"0.20"}],"y":[{"group":0,"value":"6.685600","errors":[{"label":"stat","symerror":"0.0010"},{"label":"sys","symerror":"0.20"}]},{"group":1,"value":"5.0","errors":[{"label":"stat","symerror":"0.01"},{"label":"sys","symerror":"0.2"}]}]}]})
    }
    fn contract() -> HepDataTableContract<'static> {
        HepDataTableContract {
            doi: "source/t4",
            independent_header: "PT",
            dependent_header: "yield",
            group: 0,
            qualifiers: &[("RE", "PP"), ("ENERGY", "5020")],
            required_errors: &["stat", "sys"],
        }
    }
    #[test]
    fn lexical_values_and_group_order_are_preserved() {
        let mut source = fixture();
        let expected = admit_table_json(&source.to_string(), &contract()).unwrap();
        assert_eq!(expected[0].high, "0.20");
        assert_eq!(expected[0].value, "6.685600");
        assert_eq!(expected[0].errors[0].symerror, "0.0010");
        source["values"][0]["y"].as_array_mut().unwrap().reverse();
        assert_eq!(
            admit_table_json(&source.to_string(), &contract()).unwrap(),
            expected
        );
        let mut second = contract();
        second.group = 1;
        second.qualifiers = &[("RE", "PP"), ("ENERGY", "2760")];
        assert_eq!(
            admit_table_json(&source.to_string(), &second).unwrap()[0].value,
            "5.0"
        );
    }
    #[test]
    fn identity_group_and_error_mutations_fail() {
        let source = fixture();
        for pointer in [
            "/doi",
            "/headers/1/name",
            "/qualifiers/RE/0/value",
            "/qualifiers/ENERGY/0/value",
            "/values/0/y/0/errors/0/label",
            "/values/0/y/0/value",
            "/values/0/x/0/low",
        ] {
            let mut changed = source.clone();
            *changed.pointer_mut(pointer).unwrap() = json!("wrong");
            assert!(
                admit_table_json(&changed.to_string(), &contract()).is_err(),
                "{pointer}"
            );
        }
        let mut duplicate = source.clone();
        duplicate["values"][0]["y"][1]["group"] = json!(0);
        assert!(admit_table_json(&duplicate.to_string(), &contract()).is_err());
        let mut missing = source.clone();
        missing["values"][0]["y"].as_array_mut().unwrap().remove(0);
        assert!(admit_table_json(&missing.to_string(), &contract()).is_err());
        for errors in [
            json!([]),
            json!([{"label":"stat","symerror":"0.1"}]),
            json!([{"label":"stat","symerror":"0.1"},{"label":"stat","symerror":"0.1"}]),
        ] {
            let mut changed = source.clone();
            changed["values"][0]["y"][0]["errors"] = errors;
            assert!(admit_table_json(&changed.to_string(), &contract()).is_err());
        }
    }
    #[test]
    fn spans_and_numerical_admission_fail_closed() {
        let source = fixture();
        for width in [0, 3] {
            let mut changed = source.clone();
            changed["qualifiers"]["RE"][0]["colspan"] = json!(width);
            assert!(admit_table_json(&changed.to_string(), &contract()).is_err());
        }
        let mut overlap = source.clone();
        overlap["qualifiers"]["ENERGY"][0]["colspan"] = json!(2);
        assert!(admit_table_json(&overlap.to_string(), &contract()).is_err());
        for invalid in ["NaN", "1e999", " 1", "-0.1", "-1e-999"] {
            let mut changed = source.clone();
            changed["values"][0]["y"][0]["errors"][0]["symerror"] = json!(invalid);
            assert!(admit_table_json(&changed.to_string(), &contract()).is_err());
        }
        let mut duplicate_row = source.clone();
        let row = duplicate_row["values"][0].clone();
        duplicate_row["values"].as_array_mut().unwrap().push(row);
        assert!(admit_table_json(&duplicate_row.to_string(), &contract()).is_err());
    }

    #[test]
    fn independent_units_and_sub_float_overlap_are_rejected() {
        let mut source = fixture();
        let mut expected = contract();
        expected.independent_header = "PT [GEV]";
        source["headers"][0]["name"] = json!("PT [GEV]");
        assert!(admit_table_json(&source.to_string(), &expected).is_ok());
        source["headers"][0]["name"] = json!("PT [TEV]");
        assert!(admit_table_json(&source.to_string(), &expected).is_err());
        source["headers"][0]["name"] = json!("PT [GEV]");
        let mut next = source["values"][0].clone();
        next["x"][0]["low"] = json!("0.199999999999999999999");
        next["x"][0]["high"] = json!("0.30");
        assert_eq!("0.199999999999999999999".parse::<f64>().unwrap(), 0.2);
        source["values"].as_array_mut().unwrap().push(next);
        assert!(admit_table_json(&source.to_string(), &expected).is_err());
        source["values"][1]["x"][0]["low"] = json!("2.000e-1");
        assert!(admit_table_json(&source.to_string(), &expected).is_ok());
    }

    #[test]
    fn exact_decimal_order_handles_sign_scale_and_zero() {
        for (left, right, expected) in [
            ("-1.20", "-1.19", Ordering::Less),
            ("-0.0", "0e2", Ordering::Equal),
            (".001", "1E-3", Ordering::Equal),
            ("+123e1", "1230.000", Ordering::Equal),
            ("0", "1e-20", Ordering::Less),
            ("0.200000000000000000001", ".2", Ordering::Greater),
        ] {
            assert_eq!(decimal_order(left, right).unwrap(), expected);
            assert_eq!(decimal_order(right, left).unwrap(), expected.reverse());
        }
    }

    #[test]
    fn duplicate_object_keys_are_rejected_recursively() {
        let source = fixture().to_string();
        for (original, duplicate) in [
            (
                "\"doi\":\"source/t4\"",
                "\"doi\":\"wrong\",\"doi\":\"source/t4\"",
            ),
            ("\"group\":0", "\"group\":1,\"group\":0"),
            (
                "\"value\":\"6.685600\"",
                "\"value\":\"wrong\",\"value\":\"6.685600\"",
            ),
        ] {
            assert!(source.contains(original));
            let changed = source.replacen(original, duplicate, 1);
            let error = admit_table_json(&changed, &contract()).unwrap_err();
            assert!(error.contains("duplicate JSON object key"), "{error}");
        }
    }
}
