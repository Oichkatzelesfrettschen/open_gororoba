#[cfg(test)]
use crate::construction::chingon::AlternativityViolationTensor;

#[cfg(test)]
#[test]
fn test_alternativity_violation_tensor_8d() {
    let avt = AlternativityViolationTensor::new(8);
    assert_eq!(avt.violations.len(), 0, "Octonions must be alternative");
}

#[cfg(test)]
#[test]
fn test_alternativity_violation_tensor_16d() {
    let avt = AlternativityViolationTensor::new(16);
    assert!(!avt.violations.is_empty(), "Sedenions must violate alternativity");
}

#[cfg(test)]
#[test]
fn test_alternativity_violation_tensor_64d() {
    let avt = AlternativityViolationTensor::new(64);
    assert!(!avt.violations.is_empty(), "Chingons must violate alternativity");
}
