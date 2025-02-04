use crate::filters::gaussian_state::GaussianState;
use nalgebra::{SMatrix, SVector};

pub trait SystemModel<const S: usize, const O: usize> {
    fn state_transition_matrix(&self, state: &GaussianState<S>, dt: f64) -> SMatrix<f64, S, S>;
    fn observation_matrix(&self, state: &GaussianState<S>, dt: f64) -> SMatrix<f64, O, S>;
    fn process_noise_covariance(&self, state: &GaussianState<S>, dt: f64) -> SMatrix<f64, S, S>;
    fn observation_noise_covariance(&self, state: &GaussianState<S>, dt: f64)
        -> SMatrix<f64, O, O>;

    fn propagate(&self, state: &GaussianState<S>, dt: f64) -> SVector<f64, S> {
        self.state_transition_matrix(state, dt) * state.mean
    }

    fn observe(&self, state: &GaussianState<S>, dt: f64) -> SVector<f64, O> {
        self.observation_matrix(state, dt) * state.mean
    }
}

pub struct LTISystem<const S: usize, const O: usize> {
    pub f: SMatrix<f64, S, S>, // State transition matrix
    pub h: SMatrix<f64, O, S>, // Observation Model
    pub q: SMatrix<f64, S, S>, // Process noise covariance
    pub r: SMatrix<f64, O, O>, // Observation noise covariance
}

impl<const S: usize, const O: usize> LTISystem<S, O> {
    pub fn new(
        f: SMatrix<f64, S, S>,
        h: SMatrix<f64, O, S>,
        q: SMatrix<f64, S, S>,
        r: SMatrix<f64, O, O>,
    ) -> Self {
        Self { f, h, q, r }
    }
}

impl<const S: usize, const O: usize> SystemModel<S, O> for LTISystem<S, O> {
    fn state_transition_matrix(&self, _state: &GaussianState<S>, _dt: f64) -> SMatrix<f64, S, S> {
        self.f.clone()
    }

    fn observation_matrix(&self, _state: &GaussianState<S>, _dt: f64) -> SMatrix<f64, O, S> {
        self.h.clone()
    }

    fn process_noise_covariance(&self, _state: &GaussianState<S>, _dt: f64) -> SMatrix<f64, S, S> {
        self.q.clone()
    }

    fn observation_noise_covariance(
        &self,
        _state: &GaussianState<S>,
        _dt: f64,
    ) -> SMatrix<f64, O, O> {
        self.r.clone()
    }
}
