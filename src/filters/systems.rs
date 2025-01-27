use nalgebra::SMatrix;
use crate::filters::gaussian_state::GaussianState;

pub trait SystemModel<const S: usize, const O: usize> {
    fn state_transition_matrix(&self, state: &GaussianState<S>) -> &SMatrix<f64, S, S>;
    fn observation_matrix(&self, state: &GaussianState<S>) -> &SMatrix<f64, O, S>;
    fn process_noise_covariance(&self, state: &GaussianState<S>) -> &SMatrix<f64, S, S>;
    fn observation_noise_covariance(&self, state: &GaussianState<S>) -> &SMatrix<f64, O, O>;
}

pub struct LTISystem<const S: usize, const O: usize> {
    pub F: SMatrix<f64, S, S>, // State transition matrix
    pub H: SMatrix<f64, O, S>, // Observation Model
    pub Q: SMatrix<f64, S, S>, // Process noise covariance
    pub R: SMatrix<f64, O, O>, // Observation noise covariance
}

impl<const S: usize, const O: usize> LTISystem<S, O> {
    pub fn new(F: SMatrix<f64, S, S>, H: SMatrix<f64, O, S>, Q: SMatrix<f64, S, S>, R: SMatrix<f64, O, O>) -> Self {
        Self{F, H, Q, R}
    }
}

impl<const S: usize, const O: usize> SystemModel<S, O> for LTISystem<S, O> {
    fn state_transition_matrix(&self, _state: &GaussianState<S>) -> &SMatrix<f64, S, S> {
        &self.F
    }

    fn observation_matrix(&self, _state: &GaussianState<S>) -> &SMatrix<f64, O, S> {
        &self.H
    }

    fn process_noise_covariance(&self, _state: &GaussianState<S>) -> &SMatrix<f64, S, S> {
        &self.Q
    }

    fn observation_noise_covariance(&self, _state: &GaussianState<S>) -> &SMatrix<f64, O, O> {
        &self.R
    }
}
