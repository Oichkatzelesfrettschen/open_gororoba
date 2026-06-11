use cd_kernel::cayley_dickson::cd_multiply;
use rand::Rng;

fn main() {
    let dim = 16;
    let mut rng = rand::rng();

    for _ in 0..1000 {
        let mut x: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut y: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

        // Normalize
        let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_y = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.iter_mut().for_each(|v| *v /= norm_x);
        y.iter_mut().for_each(|v| *v /= norm_y);

        let xx = cd_multiply(&x, &x);
        let xx_y = cd_multiply(&xx, &y);
        let x_y = cd_multiply(&x, &y);
        let x_xy = cd_multiply(&x, &x_y);

        let mut diff_left = 0.0;
        for k in 0..dim {
            diff_left += (xx_y[k] - x_xy[k]).powi(2);
        }

        if diff_left.sqrt() > 1e-9 {
            println!("Left alternativity failed for random sedenions!");
            return;
        }
    }
    println!("Sedenions appear to be alternative, which is wrong.");
}
