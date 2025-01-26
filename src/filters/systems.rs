use nalgebra::{SMatrix, SVector};

pub struct LTISystem<const S: usize, const O: usize> {
    pub F: SMatrix<f64, S, S>, // State transition matrix
    pub H: SMatrix<f64, O, S>, // Observation Model
    pub Q: SMatrix<f64, S, S>, // Process noise covariance
    pub R: SMatrix<f64, O, O>, // Observation noise covariance
}

