use nalgebra::{SMatrix, SVector};

pub enum KFErrors {
    UninitializedState,
}

pub struct LTISystem<const S: usize, const O: usize> {
    pub F: SMatrix<f64, S, S>, // State transition matrix
    pub H: SMatrix<f64, O, S>, // Observation Model
    pub Q: SMatrix<f64, S, S>, // Process noise covariance
    pub R: SMatrix<f64, O, O>, // Observation noise covariance
}

#[derive(Clone)]
pub struct StateEstimate<const S: usize> {
    pub state: SVector<f64, S>,
    pub covariance: SMatrix<f64, S, S>
}

pub struct KalmanFilter<const S: usize, const O: usize> {
    system_model: LTISystem<S, O>,
    posterior: Option<StateEstimate<S>>,
}

impl<const S: usize, const O: usize> KalmanFilter<S, O> {
    pub fn new(system_model: LTISystem<S, O>) -> Self{
        Self{system_model, posterior: None}
    }

    pub fn predict(&mut self) -> Result<StateEstimate<S>, KFErrors> {
        match &self.posterior {
            Some(p) => Ok(p.clone()),
            None => Err(KFErrors::UninitializedState),
        }
    }

    pub fn update(&mut self, z: &SVector<f64, O>) {
        let prior = match &self.posterior {
            Some(posterior) => StateEstimate{
                state: self.system_model.F * posterior.state,
                covariance: self.system_model.F * posterior.covariance * self.system_model.F.transpose() + self.system_model.Q,
            },
            None => StateEstimate {
                state: SVector::zeros(),
                covariance: SMatrix::identity(),
            }
        };

        let innovation = StateEstimate::<O> {
            state: z - self.system_model.H * prior.state,
            covariance: self.system_model.H * prior.covariance * self.system_model.H.transpose() + self.system_model.R,
        };

        let kalman_gain = match innovation.covariance.try_inverse() {
            Some(inv) => prior.covariance * self.system_model.H.transpose() * inv,
            None => panic!(),
        };

        self.posterior = Some(StateEstimate {
            state: prior.state + kalman_gain * innovation.state,
            covariance: (SMatrix::identity() - kalman_gain * self.system_model.H) * prior.covariance,
        });
    }
}
