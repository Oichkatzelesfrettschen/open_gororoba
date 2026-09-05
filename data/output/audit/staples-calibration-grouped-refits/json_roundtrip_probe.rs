fn main() {
    let mut state = 42u64;
    let mut mismatches = 0usize;
    for _ in 0..100000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let number = state as f64 / u64::MAX as f64;
        if !number.is_finite() { continue; }
        let value = serde_json::json!({"value":number});
        let serialized = serde_json::to_vec(&value).unwrap();
        let restored: serde_json::Value = serde_json::from_slice(&serialized).unwrap();
        let reserialized = serde_json::to_vec(&restored).unwrap();
        if serialized != reserialized {
            if mismatches == 0 { println!("first mismatch: {} -> {}", String::from_utf8_lossy(&serialized), String::from_utf8_lossy(&reserialized)); }
            mismatches += 1;
        }
    }
    println!("mismatches: {mismatches}");
}
