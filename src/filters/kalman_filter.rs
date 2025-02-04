use crate::filters::gaussian_state::GaussianState;
use crate::filters::systems::SystemModel;
use nalgebra::{SMatrix, SVector};

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

pub struct KalmanFilter<'a, const S: usize, const O: usize> {
    system_model: Box<dyn SystemModel<S, O> + 'a>,
    posterior: Option<GaussianState<S>>,
}

impl<'a, const S: usize, const O: usize> KalmanFilter<'a, S, O> {
    pub fn new(system_model: impl SystemModel<S, O> + 'a) -> Self {
        Self {
            system_model: Box::new(system_model),
            posterior: None,
        }
    }

    pub fn predict(&mut self) -> Result<GaussianState<S>, KFError> {
        match &self.posterior {
            Some(p) => Ok(p.clone()),
            None => Err(KFError::UninitializedState),
        }
    }

    pub fn update(&mut self, z: &SVector<f64, O>, dt: f64) -> Result<(), KFError> {
        let posterior = match &self.posterior {
            Some(posterior) => posterior,
            None => &GaussianState {
                mean: SVector::zeros(),
                covariance: SMatrix::identity(),
            },
        };

        let f = self.system_model.state_transition_matrix(posterior, dt);
        let q = self.system_model.process_noise_covariance(posterior, dt);

        let prior = GaussianState {
            mean: self.system_model.propagate(posterior, dt),
            covariance: f * posterior.covariance * f.transpose() + q,
        };

        let h = self.system_model.observation_matrix(&prior, dt);
        let r = self.system_model.observation_noise_covariance(&prior, dt);

        let innovation = GaussianState {
            mean: z - self.system_model.observe(&prior, dt),
            covariance: h * prior.covariance * h.transpose() + r,
        };

        let kalman_gain = match innovation.covariance.try_inverse() {
            Some(inv) => prior.covariance * h.transpose() * inv,
            None => panic!(),
        };

        self.posterior = Some(GaussianState {
            mean: prior.mean + kalman_gain * innovation.mean,
            covariance: (SMatrix::identity() - kalman_gain * h) * prior.covariance,
        });

        Ok(())
    }
}
