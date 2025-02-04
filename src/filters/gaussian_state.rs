use nalgebra::{SMatrix, SVector};

#[derive(Clone)]
pub struct GaussianState<const S: usize> {
    pub mean: SVector<f64, S>,
    pub covariance: SMatrix<f64, S, S>,
}
