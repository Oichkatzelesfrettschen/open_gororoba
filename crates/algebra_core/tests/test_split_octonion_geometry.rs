use algebra_core::construction::cayley_dickson::{cd_multiply, cd_multiply_split, CdSignature};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SimpleBlade {
    i: usize,
    j: usize,
    sign: i8,
}

impl SimpleBlade {
    fn new(i: usize, j: usize, sign: i8) -> Self {
        assert!(i < j);
        Self { i, j, sign }
    }

    fn to_vec(self) -> Vec<f64> {
        let mut v = vec![0.0; 8];
        v[self.i] = 1.0;
        v[self.j] = self.sign as f64;
        v
    }

    fn to_vec_16(self) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        v[self.i] = 1.0;
        v[self.j] = self.sign as f64;
        v
    }
}

#[test]
fn test_split_octonion_basis_squares() {
    let dim = 8;
    let sig = CdSignature::split(dim);
    let mut squares = Vec::new();
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        let sq = cd_multiply_split(&v, &v, &sig);
        squares.push(sq[0] as i32);
    }
    assert_eq!(squares, vec![1, 1, 1, -1, 1, -1, -1, 1]);
}

#[test]
fn test_split_octonion_zero_divisor_census() {
    let dim = 8;
    let sig = CdSignature::split(dim);
    let mut blades = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            blades.push(SimpleBlade::new(i, j, 1));
            blades.push(SimpleBlade::new(i, j, -1));
        }
    }

    let mut zd_pairs = Vec::new();
    for (idx1, b1) in blades.iter().enumerate() {
        for (idx2, b2) in blades.iter().enumerate() {
            if idx1 >= idx2 {
                continue;
            }
            let prod = cd_multiply_split(&b1.to_vec(), &b2.to_vec(), &sig);
            if prod.iter().map(|x| x * x).sum::<f64>() < 1e-10 {
                let sq1: f64 = cd_multiply_split(&b1.to_vec(), &b1.to_vec(), &sig)
                    .iter()
                    .map(|x| x * x)
                    .sum();
                let sq2: f64 = cd_multiply_split(&b2.to_vec(), &b2.to_vec(), &sig)
                    .iter()
                    .map(|x| x * x)
                    .sum();
                zd_pairs.push((*b1, *b2, sq1 < 1e-10, sq2 < 1e-10));
            }
        }
    }
    assert_eq!(zd_pairs.len(), 52);
}

#[test]
fn test_split_octonion_sign_census() {
    let dim = 8;
    let sig = CdSignature::split(dim);
    let mut neg_count = 0;
    for i in 0..dim {
        for j in 0..dim {
            let v_i = {
                let mut v = vec![0.0; dim];
                v[i] = 1.0;
                v
            };
            let v_j = {
                let mut v = vec![0.0; dim];
                v[j] = 1.0;
                v
            };
            let prod = cd_multiply_split(&v_i, &v_j, &sig);
            if prod.iter().any(|&x| x < -0.1) {
                neg_count += 1;
            }
        }
    }
    assert_eq!(neg_count, 24);
}

#[test]
fn test_null_cloud_topology() {
    let dim = 8;
    let sig = CdSignature::split(dim);
    let mut blades = Vec::new();
    for i in 0..dim {
        for j in (i + 1)..dim {
            blades.push(SimpleBlade::new(i, j, 1));
            blades.push(SimpleBlade::new(i, j, -1));
        }
    }
    let mut nn_count = 0;
    for (idx1, b1) in blades.iter().enumerate() {
        for (idx2, b2) in blades.iter().enumerate() {
            if idx1 >= idx2 {
                continue;
            }
            let prod = cd_multiply_split(&b1.to_vec(), &b2.to_vec(), &sig);
            if prod.iter().map(|x| x * x).sum::<f64>() < 1e-10 {
                let sq1: f64 = cd_multiply_split(&b1.to_vec(), &b1.to_vec(), &sig)
                    .iter()
                    .map(|x| x * x)
                    .sum();
                let sq2: f64 = cd_multiply_split(&b2.to_vec(), &b2.to_vec(), &sig)
                    .iter()
                    .map(|x| x * x)
                    .sum();
                if sq1 < 1e-10 && sq2 < 1e-10 {
                    nn_count += 1;
                }
            }
        }
    }
    assert_eq!(nn_count, 24);
}

#[test]
fn test_hybrid_16_frustration() {
    let dim = 16;
    let gammas = vec![1, 1, 1, -1];
    let sig_hybrid = CdSignature::from_gammas(&gammas);
    let mut neg_count = 0;
    for i in 0..dim {
        for j in 0..dim {
            let v_i = {
                let mut v = vec![0.0; dim];
                v[i] = 1.0;
                v
            };
            let v_j = {
                let mut v = vec![0.0; dim];
                v[j] = 1.0;
                v
            };
            let prod = cd_multiply_split(&v_i, &v_j, &sig_hybrid);
            if prod.iter().any(|&x| x < -0.1) {
                neg_count += 1;
            }
        }
    }
    let ratio = neg_count as f64 / 256.0;
    println!(
        "Hybrid (1,1,1,-1) Sign Ratio: {}/256 = {}",
        neg_count, ratio
    );

    let sig_std = CdSignature::from_gammas(&[-1, -1, -1, -1]);
    let mut neg_std = 0;
    for i in 0..dim {
        for j in 0..dim {
            let v_i = {
                let mut v = vec![0.0; dim];
                v[i] = 1.0;
                v
            };
            let v_j = {
                let mut v = vec![0.0; dim];
                v[j] = 1.0;
                v
            };
            let prod = cd_multiply_split(&v_i, &v_j, &sig_std);
            if prod.iter().any(|&x| x < -0.1) {
                neg_std += 1;
            }
        }
    }
    println!(
        "Standard Sedenion Sign Ratio: {}/256 = {}",
        neg_std,
        neg_std as f64 / 256.0
    );
}

#[test]
fn test_unit_census() {
    let dim16 = 16;
    let mut units16 = std::collections::HashSet::new();
    for i in 0..dim16 {
        for j in (i + 1)..dim16 {
            for k in 0..dim16 {
                for l in (k + 1)..dim16 {
                    if i == k || i == l || j == k || j == l {
                        continue;
                    }
                    let v1 = SimpleBlade::new(i, j, 1).to_vec_16();
                    let v2 = SimpleBlade::new(k, l, 1).to_vec_16();
                    let prod = cd_multiply(&v1, &v2);
                    if prod.iter().map(|x| x * x).sum::<f64>() < 1e-10 {
                        let mut set = vec![i, j, k, l];
                        set.sort();
                        units16.insert(set);
                    }
                }
            }
        }
    }
    assert_eq!(units16.len(), 42);
}
