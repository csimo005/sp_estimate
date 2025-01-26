use std::collections::VecDeque;
use plotters::prelude::*;
use rand::distributions::Distribution;
use rand::thread_rng;
use statrs::distribution::Normal;
use nalgebra::*;

use sp_estimate::filters::kalman_filter::KalmanFilter;
use sp_estimate::filters::systems::LTISystem;

const ANIMATION_FILE_NAME: &str = "animation.gif";
const ERROR_CHART_NAME: &str = "error.png";

const FREQ: f64 = 0.7854;
const T_MAX: f64 = 20.0;
const DT: f64 = 0.05;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let n = Normal::new(0.0, 0.1).unwrap();

    let mut t: f64 = 0.0;

    let mut data: VecDeque<(f64, f64)> = VecDeque::<(f64, f64)>::new();
    let mut obsv: VecDeque<(f64, f64)> = VecDeque::<(f64, f64)>::new();
    let mut obsv_error = Vec::<f64>::new();
    let mut kf_est: VecDeque<(f64, f64)> = VecDeque::<(f64, f64)>::new();
    let mut kf_error = Vec::<f64>::new();

    let sys = LTISystem {
        F: matrix!{
            1.0, 0.0, DT, 0.0, 0.5 * DT.powf(2.0), 0.0;
            0.0, 1.0, 0.0, DT, 0.0, 0.5 * DT.powf(2.0);
            0.0, 0.0, 1.0, 0.0, DT, 0.0;
            0.0, 0.0, 0.0, 1.0, 0.0, DT;
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        },
        H: matrix!{
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        },
        Q: 0.001 * SMatrix::identity(),
        R: 0.1 * SMatrix::identity(),
    };
    let mut KF = KalmanFilter::new(sys);

    let root = BitMapBackend::gif(ANIMATION_FILE_NAME, (400, 400), (1000. * DT) as u32)?.into_drawing_area();
    while t < T_MAX {
        // Step system
        let state = ((FREQ * t).cos(), (FREQ * t).sin());
        data.push_front(state);
        if data.len() > 20 {
            data.pop_back();
        }

        let w: Vec<_> = n.sample_iter(&mut rng).take(2).collect();
        obsv.push_front((state.0 + w[0], state.1 + w[1]));
        if obsv.len() > 20 {
            obsv.pop_back();
        }
        obsv_error.push((w[0].powf(2.0) + w[1].powf(2.0)).sqrt());

        match obsv.front() {
            Some(p) => KF.update(&vector!{p.0, p.1}),
            None => unreachable!(),
        }

        let est = match KF.predict() {
            Ok(e) => e,
            Err(_) => panic!(),
        };

        kf_est.push_front((est.mean[0], est.mean[1]));
        if kf_est.len() > 20 {
            kf_est.pop_back();
        }
        kf_error.push(((est.mean[0] - state.0).powf(2.0) + (est.mean[1] - state.1).powf(2.0)).sqrt());

        t += DT;

        // Render
        root.fill(&BLACK)?;

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(-2.0..2.0, -2.0..2.0)?;

        chart.draw_series(LineSeries::new(data.clone(), &RED))?;
        match data.front() {
            Some(p) => chart.draw_series(vec![Circle::new((p.0, p.1), 2, RED.filled())])?,
            None => unreachable!(),
        };
        
        //chart.draw_series(LineSeries::new(obsv.clone(), &YELLOW))?;
        match obsv.front() {
            Some(p) => chart.draw_series(vec![Circle::new((p.0, p.1), 2, YELLOW.filled())])?,
            None => unreachable!(),
        };
        
        chart.draw_series(LineSeries::new(kf_est.clone(), &GREEN))?;
        match kf_est.front() {
            Some(p) => chart.draw_series(vec![Circle::new((p.0, p.1), 2, GREEN.filled())])?,
            None => unreachable!(),
        };

        root.present()?;
    }

    let root = BitMapBackend::new(ERROR_CHART_NAME, (600, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("KF Error", ("sans-serif", 40))
        .build_cartesian_2d(0f64..T_MAX, 0f64..0.5f64)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();
    ctx.draw_series(
      LineSeries::new((0..kf_error.len()).map(|i| ((i as f64) * DT, kf_error[i])), &BLUE)
    ).unwrap();
    ctx.draw_series(
      LineSeries::new((0..kf_error.len()).map(|i| ((i as f64) * DT, obsv_error[i])), &RED)
    ).unwrap();

    Ok(())
}

#[test]
fn kalman_filter_test() {
    main().unwrap()
}
