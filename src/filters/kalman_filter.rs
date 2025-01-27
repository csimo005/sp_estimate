use nalgebra::{SMatrix, SVector};
use crate::filters::gaussian_state::GaussianState;
use crate::filters::systems::{LTISystem,SystemModel};

#[derive(Debug)]
pub enum KFError {
    UninitializedState,
}

impl std::fmt::Display for KFError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            KFError::UninitializedState => write!(f, "No measurements recieved"),
            // Add more match arms as needed
        }
    }
}

impl std::error::Error for KFError {
    fn description(&self) -> &str {
        match *self {
            KFError::UninitializedState => "No measurements recieved",
            // Add more match arms as needed
        }
    }
}

pub struct KalmanFilter<const S: usize, const O: usize> {
    system_model: LTISystem<S, O>,
    posterior: Option<GaussianState<S>>,
}

impl<const S: usize, const O: usize> KalmanFilter<S, O> {
    pub fn new(system_model: LTISystem<S, O>) -> Self{
        Self{system_model, posterior: None}
    }

    pub fn predict(&mut self) -> Result<GaussianState<S>, KFError> {
        match &self.posterior {
            Some(p) => Ok(p.clone()),
            None => Err(KFError::UninitializedState),
        }
    }

    pub fn update(&mut self, z: &SVector<f64, O>) -> Result<(), KFError>{
        let posterior = match &self.posterior {
            Some(posterior) => posterior,
            None => &GaussianState {
                mean: SVector::zeros(),
                covariance: SMatrix::identity(),
            },
        };

        let F = self.system_model.state_transition_matrix(posterior);
        let H = self.system_model.observation_matrix(posterior);
        let Q = self.system_model.process_noise_covariance(posterior);
        let R = self.system_model.observation_noise_covariance(posterior);

        let prior = GaussianState{
                mean: F * posterior.mean,
                covariance: F * posterior.covariance * F.transpose() + Q,
        };

        let innovation = GaussianState {
            mean: z - H * prior.mean,
            covariance: H * prior.covariance * H.transpose() + R,
        };

        let kalman_gain = match innovation.covariance.try_inverse() {
            Some(inv) => prior.covariance * H.transpose() * inv,
            None => panic!(),
        };

        self.posterior = Some(GaussianState {
            mean: prior.mean + kalman_gain * innovation.mean,
            covariance: (SMatrix::identity() - kalman_gain * H) * prior.covariance,
        });

        Ok(())
    }
}
