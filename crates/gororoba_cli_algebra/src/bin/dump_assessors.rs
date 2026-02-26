use algebra_core::analysis::boxkites::primitive_assessors;

fn main() {
    let assessors = primitive_assessors();
    // We need pairs for 84 ZDs. The 42 assessors give (a,b).
    // The ZDs are (a,b, +1) and (a,b, -1).
    // We can just list the 42 pairs and handle signs in shader loop.
    print!("const uint assessors[84] = uint[](");
    for a in &assessors {
        print!("{},{},", a.low, a.high);
    }
    println!(");");
}
