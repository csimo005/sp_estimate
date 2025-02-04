use nalgebra::*;
use rand::distributions::Distribution;
use rand::thread_rng;
use statrs::distribution::Normal;

use sp_estimate::filters::gaussian_state::GaussianState;
use sp_estimate::filters::kalman_filter::KalmanFilter;
use sp_estimate::filters::systems::{LTISystem, SystemModel};

mod utils;

const FREQ: f64 = 0.7854;
const T_MAX: f64 = 20.0;
const DT: f64 = 0.05;

fn generate_states() -> Vec<(f64, f64)> {
    let mut states = Vec::<(f64, f64)>::new();
    let mut t: f64 = 0.;

    while t < T_MAX {
        states.push(((FREQ * t).cos(), (FREQ * t).sin()));
        t += DT;
    }

    states
}

fn test_filter<const S: usize>(
    animation_fname: &str,
    error_plot_fname: &str,
    filter: &mut KalmanFilter<S, 2>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let n = Normal::new(0.0, 0.1).unwrap();

    let states = generate_states();

    let obsv: Vec<_> = states
        .iter()
        .map(|s| {
            let w: Vec<_> = n.sample_iter(&mut rng).take(2).collect();
            (s.0 + w[0], s.1 + w[1])
        })
        .collect();
    let obsv_error: Vec<f64> = states
        .iter()
        .zip(obsv.iter())
        .map(|(a, b)| ((a.0 - b.0).powf(2.0) + (a.1 - b.1).powf(2.0)).sqrt())
        .collect();

    let est: Vec<_> = obsv
        .iter()
        .map(|o| {
            let _ = filter.update(&vector! {o.0, o.1}, DT);
            let est = match filter.predict() {
                Ok(e) => e,
                Err(_) => panic!(),
            };

            (est.mean[0], est.mean[1])
        })
        .collect();
    let est_error: Vec<f64> = states
        .iter()
        .zip(est.iter())
        .map(|(a, b)| ((a.0 - b.0).powf(2.0) + (a.1 - b.1).powf(2.0)).sqrt())
        .collect();

    utils::animate_states_2d(animation_fname, (DT * 1000.) as u32, states, obsv, est)?;
    utils::plot_error(error_plot_fname, DT, obsv_error, est_error)?;
    Ok(())
}

struct XYThetaTracker {
    h: SMatrix<f64, 2, 5>,
    q: SMatrix<f64, 5, 5>,
    r: SMatrix<f64, 2, 2>,
}

impl XYThetaTracker {
    fn new(process_noise: f64, observation_noise: f64) -> Self {
        let q = process_noise * SMatrix::<f64, 5, 5>::identity();
        let r = observation_noise * SMatrix::<f64, 2, 2>::identity();
        Self {
            h: SMatrix::identity(),
            q,
            r,
        }
    }
}

impl SystemModel<5, 2> for XYThetaTracker {
    fn propagate(&self, state: &GaussianState<5>, dt: f64) -> SVector<f64, 5> {
        let mut next: SVector<f64, 5> = SVector::zeros();
        next[0] = state.mean[0] + dt * state.mean[3] * state.mean[2].cos();
        next[1] = state.mean[1] + dt * state.mean[3] * state.mean[2].sin();
        next[2] = state.mean[2] + dt * state.mean[4];
        next[3] = state.mean[3];
        next[4] = state.mean[4];

        next
    }

    fn observe(&self, state: &GaussianState<5>, _dt: f64) -> SVector<f64, 2> {
        self.h * state.mean
    }

    fn state_transition_matrix(&self, state: &GaussianState<5>, dt: f64) -> SMatrix<f64, 5, 5> {
        let mut f: SMatrix<f64, 5, 5> = SMatrix::identity();
        f[(0, 2)] = -1. * dt * state.mean[3] * state.mean[2].sin();
        f[(0, 3)] = dt * state.mean[2].cos();
        f[(1, 2)] = dt * state.mean[3] * state.mean[2].cos();
        f[(1, 3)] = dt * state.mean[2].sin();
        f[(2, 4)] = dt;

        f
    }

    fn observation_matrix(&self, _state: &GaussianState<5>, _dt: f64) -> SMatrix<f64, 2, 5> {
        self.h.clone()
    }

    fn process_noise_covariance(&self, _state: &GaussianState<5>, _dt: f64) -> SMatrix<f64, 5, 5> {
        self.q.clone()
    }

    fn observation_noise_covariance(
        &self,
        _state: &GaussianState<5>,
        _dt: f64,
    ) -> SMatrix<f64, 2, 2> {
        self.r.clone()
    }
}

#[test]
fn xytheta_ekf() -> Result<(), Box<dyn std::error::Error>> {
    let sys = XYThetaTracker::new(0.001, 0.1);
    let mut kf = KalmanFilter::new(sys);
    test_filter("xytheta_animation.gif", "xytheta_error.png", &mut kf)?;

    Ok(())
}

#[test]
fn constant_velocity_kf() -> Result<(), Box<dyn std::error::Error>> {
    let sys = LTISystem::new(
        matrix! {
            1.0, 0.0, DT, 0.0;
            0.0, 1.0, 0.0, DT;
            0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0;
        },
        matrix! {
            1.0, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0;
        },
        0.001 * SMatrix::identity(),
        0.1 * SMatrix::identity(),
    );
    let mut kf = KalmanFilter::new(sys);
    test_filter(
        "constant_velocity_animation.gif",
        "constant_velocity_error.png",
        &mut kf,
    )?;

    Ok(())
}

#[test]
fn constant_acceleration_kf() -> Result<(), Box<dyn std::error::Error>> {
    let sys = LTISystem::new(
        matrix! {
            1.0, 0.0, DT, 0.0, 0.5 * DT.powf(2.0), 0.0;
            0.0, 1.0, 0.0, DT, 0.0, 0.5 * DT.powf(2.0);
            0.0, 0.0, 1.0, 0.0, DT, 0.0;
            0.0, 0.0, 0.0, 1.0, 0.0, DT;
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        },
        matrix! {
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        },
        0.001 * SMatrix::identity(),
        0.1 * SMatrix::identity(),
    );
    let mut kf = KalmanFilter::new(sys);
    test_filter(
        "constant_acceleration_animation.gif",
        "constant_acceleration_error.png",
        &mut kf,
    )?;

    Ok(())
}
