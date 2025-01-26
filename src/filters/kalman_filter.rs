use nalgebra::{SMatrix, SVector};
use crate::filters::gaussian_state::GaussianState;
use crate::filters::systems::LTISystem;

pub enum KFErrors {
    UninitializedState,
}

pub struct KalmanFilter<const S: usize, const O: usize> {
    system_model: LTISystem<S, O>,
    posterior: Option<GaussianState<S>>,
}

impl<const S: usize, const O: usize> KalmanFilter<S, O> {
    pub fn new(system_model: LTISystem<S, O>) -> Self{
        Self{system_model, posterior: None}
    }

    pub fn predict(&mut self) -> Result<GaussianState<S>, KFErrors> {
        match &self.posterior {
            Some(p) => Ok(p.clone()),
            None => Err(KFErrors::UninitializedState),
        }
    }

    pub fn update(&mut self, z: &SVector<f64, O>) {
        let prior = match &self.posterior {
            Some(posterior) => GaussianState{
                mean: self.system_model.F * posterior.mean,
                covariance: self.system_model.F * posterior.covariance * self.system_model.F.transpose() + self.system_model.Q,
            },
            None => GaussianState {
                mean: SVector::zeros(),
                covariance: SMatrix::identity(),
            }
        };

        let innovation = GaussianState::<O> {
            mean: z - self.system_model.H * prior.mean,
            covariance: self.system_model.H * prior.covariance * self.system_model.H.transpose() + self.system_model.R,
        };

        let kalman_gain = match innovation.covariance.try_inverse() {
            Some(inv) => prior.covariance * self.system_model.H.transpose() * inv,
            None => panic!(),
        };

        self.posterior = Some(GaussianState {
            mean: prior.mean + kalman_gain * innovation.mean,
            covariance: (SMatrix::identity() - kalman_gain * self.system_model.H) * prior.covariance,
        });
    }
}
