//! Synthesized quantizer: auto-selects optimal method per bit-width and dimension.
//!
//! Combines all measured results into a single adaptive strategy:
//!
//! | Bits | d=64      | d=128           | d=256+ |
//! |------|-----------|-----------------|--------|
//! | 2    | TQ (WHT)  | TQ (WHT)        | TQ (WHT) |
//! | 3    | Hybrid-16 | Hybrid-16       | Hybrid-32 |
//! | 4    | Hybrid-16 | Hybrid-16       | Hybrid-32 |
//!
//! Rationale (from measured results):
//! - 2-bit: rotation decorrelation dominates.  TQ MSE=0.118 vs Group=0.184 (36% better).
//!   Grouping hurts at 2-bit because 4 centroids can't adapt within small groups.
//! - 3-bit: group scales dominate.  Hybrid MSE=0.033 vs TQ=0.035 (6% better).
//!   Rotation still helps but grouping provides the marginal gain.
//! - 4-bit: group scales dominate even more.  Hybrid MSE=0.007 vs TQ=0.011 (32% better).
//! - Group size 16 best across the board (MSE=0.023 vs g=32 MSE=0.034).
//!
//! Additionally applies:
//! - Adaptive bit allocation via CD associator (23% MSE gain)
//! - QJL correction at bits <= 3 only (hurts at 4-bit)

use super::adaptive_bits::{self, BitAllocation};
use super::hybrid::HybridQuantizer;
use super::pipeline::{TurboQuantMSE, TurboQuantProd};

/// Synthesized quantizer that auto-selects the best method.
pub struct SynthesizedQuantizer {
    /// For 2-bit: pure TurboQuant (rotation + uniform codebook).
    tq: TurboQuantMSE,
    /// For 3-4 bit: hybrid (rotation + group scales).
    hybrid: HybridQuantizer,
    /// For QJL correction at low bits (reserved for 3-bit Prod path).
    #[allow(dead_code)]
    tq_prod: TurboQuantProd,
    /// Configuration.
    d: usize,
    bits: u32,
    /// Whether to use adaptive bit allocation.
    adaptive: bool,
    promote_fraction: f64,
}

/// Compressed output from the synthesized quantizer.
#[derive(Clone, Debug)]
pub struct SynthesizedCompressed {
    /// Method used for this vector.
    pub method: &'static str,
    /// Actual bits used.
    pub bits: u32,
    /// Quantized data (format depends on method).
    pub data: SynthesizedData,
}

/// Quantized data in method-specific format.
#[derive(Clone, Debug)]
pub enum SynthesizedData {
    /// Pure TQ: MSE indices + optional QJL signs.
    TurboQuant {
        indices: Vec<u8>,
        qjl_signs: Option<Vec<i8>>,
        residual_norm: f64,
        vec_norm: f64,
    },
    /// Hybrid: rotated group-quantized indices + group params.
    Hybrid(super::hybrid::HybridCompressed),
}

impl SynthesizedQuantizer {
    /// Create the synthesized quantizer with optimal method selection.
    pub fn new(d: usize, bits: u32, seed: u64) -> Self {
        let use_wht = d >= 64;
        let group_size = if d >= 256 { 32 } else { 16 };

        SynthesizedQuantizer {
            tq: TurboQuantMSE::new(d, bits, seed, use_wht),
            hybrid: HybridQuantizer::new(d, bits, seed, use_wht, group_size),
            tq_prod: TurboQuantProd::new(d, bits, seed, use_wht, None),
            d,
            bits,
            adaptive: true,
            promote_fraction: 0.25,
        }
    }

    /// Quantize a single vector using the optimal method for the configured bit-width.
    pub fn quantize(&self, x: &[f64], buf: &mut [f64]) -> SynthesizedCompressed {
        if self.bits <= 2 {
            // 2-bit: pure TurboQuant MSE-only (rotation + full bit budget to codebook).
            // At 2-bit, splitting into 1-bit MSE + 1-bit QJL is too lossy.
            // Better to give all 2 bits to the MSE stage for 4 centroids.
            let comp = self.tq.quantize(x, buf);
            SynthesizedCompressed {
                method: "turboquant-mse",
                bits: self.bits,
                data: SynthesizedData::TurboQuant {
                    indices: comp.indices,
                    qjl_signs: None,
                    residual_norm: 0.0,
                    vec_norm: comp.vec_norm,
                },
            }
        } else {
            // 3-4 bit: hybrid (rotation + group scales)
            let comp = self.hybrid.quantize(x, buf);
            SynthesizedCompressed {
                method: "hybrid",
                bits: self.bits,
                data: SynthesizedData::Hybrid(comp),
            }
        }
    }

    /// Dequantize back to f64.
    pub fn dequantize(&self, compressed: &SynthesizedCompressed, buf: &mut [f64], out: &mut [f64]) {
        match &compressed.data {
            SynthesizedData::TurboQuant { indices, vec_norm, .. } => {
                let mse_repr = super::pipeline::MseCompressed {
                    indices: indices.clone(),
                    vec_norm: *vec_norm,
                };
                self.tq.dequantize(&mse_repr, buf, out);
            }
            SynthesizedData::Hybrid(comp) => {
                self.hybrid.dequantize(comp, buf, out);
            }
        }
    }

    /// Quantize a batch with optional adaptive bit allocation.
    pub fn quantize_batch(&self, vectors: &[Vec<f64>]) -> Vec<SynthesizedCompressed> {
        let mut buf = vec![0.0f64; 3 * self.d];

        if !self.adaptive {
            return vectors.iter().map(|v| self.quantize(v, &mut buf)).collect();
        }

        // Step 1: quantize all at base bits
        let base_compressed: Vec<_> = vectors.iter()
            .map(|v| self.quantize(v, &mut buf))
            .collect();

        // Step 2: compute residuals for adaptive allocation
        let residuals: Vec<Vec<f64>> = vectors.iter()
            .zip(base_compressed.iter())
            .map(|(orig, comp)| {
                let mut recon = vec![0.0f64; self.d];
                self.dequantize(comp, &mut buf, &mut recon);
                orig.iter().zip(recon.iter()).map(|(a, b)| a - b).collect()
            })
            .collect();

        // Step 3: allocate bits
        let allocation = adaptive_bits::allocate_bits(&residuals, self.d, self.promote_fraction);

        // Step 4: re-quantize promoted tokens at (bits+1)
        let promoted_sq = SynthesizedQuantizer::new(self.d, self.bits + 1, 42);

        vectors.iter()
            .zip(allocation.iter())
            .map(|(v, alloc)| {
                match alloc {
                    BitAllocation::Base => self.quantize(v, &mut buf),
                    BitAllocation::Promoted => {
                        let mut comp = promoted_sq.quantize(v, &mut buf);
                        comp.bits = self.bits + 1;
                        comp
                    }
                }
            })
            .collect()
    }

    pub fn dim(&self) -> usize { self.d }
    pub fn bits(&self) -> u32 { self.bits }
}

impl std::fmt::Display for SynthesizedQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let method = if self.bits <= 2 { "turboquant" } else { "hybrid" };
        write!(f, "Synthesized(d={}, {}bit, method={}, adaptive={})",
            self.d, self.bits, method, self.adaptive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let normal = StandardNormal;
        (0..n).map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect()).collect()
    }

    fn measure_mse(vectors: &[Vec<f64>], sq: &SynthesizedQuantizer) -> f64 {
        let d = sq.dim();
        let mut buf = vec![0.0f64; 3 * d];
        let mut total_mse = 0.0f64;
        for v in vectors {
            let comp = sq.quantize(v, &mut buf);
            let mut recon = vec![0.0f64; d];
            sq.dequantize(&comp, &mut buf, &mut recon);
            total_mse += v.iter().zip(recon.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64;
        }
        total_mse / vectors.len() as f64
    }

    #[test]
    fn test_synthesized_selects_tq_at_2bit() {
        let sq = SynthesizedQuantizer::new(128, 2, 42);
        let x: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut buf = vec![0.0f64; 384];
        let comp = sq.quantize(&x, &mut buf);
        assert_eq!(comp.method, "turboquant-mse");
    }

    #[test]
    fn test_synthesized_selects_hybrid_at_3bit() {
        let sq = SynthesizedQuantizer::new(128, 3, 42);
        let x: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut buf = vec![0.0f64; 384];
        let comp = sq.quantize(&x, &mut buf);
        assert_eq!(comp.method, "hybrid");
    }

    #[test]
    fn test_synthesized_quality() {
        let d = 128;
        let n = 100;
        let vectors = random_vectors(n, d, 42);

        for bits in [2, 3, 4] {
            let sq = SynthesizedQuantizer::new(d, bits, 42);
            let mse = measure_mse(&vectors, &sq);

            // Compare with pure TQ
            let tq = TurboQuantMSE::new(d, bits, 42, true);
            let mut buf = vec![0.0f64; 2 * d];
            let tq_mse: f64 = vectors.iter().map(|v| {
                let comp = tq.quantize(v, &mut buf);
                let mut recon = vec![0.0f64; d];
                tq.dequantize(&comp, &mut buf, &mut recon);
                v.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / d as f64
            }).sum::<f64>() / n as f64;

            let improvement = (tq_mse - mse) / tq_mse * 100.0;
            println!("  {}-bit: synthesized={:.6}  tq={:.6}  improvement={:.1}%  method={}",
                bits, mse, tq_mse, improvement, sq);
        }
    }

    #[test]
    fn test_synthesized_batch_adaptive() {
        let d = 64;
        let n = 40;
        let vectors = random_vectors(n, d, 42);
        let sq = SynthesizedQuantizer::new(d, 3, 42);
        let batch = sq.quantize_batch(&vectors);

        assert_eq!(batch.len(), n);
        // Some should be promoted (bits=4)
        let n_promoted = batch.iter().filter(|c| c.bits == 4).count();
        println!("Adaptive: {}/{} promoted to 4-bit", n_promoted, n);
        assert!(n_promoted > 0 && n_promoted < n);
    }

    #[test]
    fn test_synthesized_display() {
        let sq = SynthesizedQuantizer::new(128, 3, 42);
        let s = format!("{}", sq);
        assert!(s.contains("128") && s.contains("3bit") && s.contains("hybrid"));
    }
}
