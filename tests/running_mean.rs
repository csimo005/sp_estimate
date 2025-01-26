use plotters::prelude::*;
use rand::distributions::Distribution;
use rand::thread_rng;
use sp_estimate::parameter::running_mean::RunningMean;
use statrs::distribution::Normal;

#[test]
fn running_mean_test() -> Result<(), Box<dyn std::error::Error>> {
    const OUT_FILE_NAME: &str = "running_mean.png";
    const N_SAMPLES: usize = 500;

    let param: f32 = 3.14159;
    let n = Normal::new(0.0, 1.0).unwrap();

    let mut rng = thread_rng();
    let data: Vec<_> = n
        .sample_iter(&mut rng)
        .take(N_SAMPLES)
        .map(|w| param + (w as f32))
        .collect();

    let mut est: Vec<_> = Vec::<(f32, f32)>::new();
    let mut cnf: Vec<_> = Vec::<f32>::new();
    let mut rm = RunningMean::new();
    for i in 0..data.len() {
        rm.update(data[i]);
        match rm.get() {
            Ok((mean, var)) => est.push((mean, var)),
            Err(_) => (),
        };
        match rm.confidence(0.95f32) {
            Ok(c) => cnf.push(c),
            Err(_) => (),
        };
    }

    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(100)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..(N_SAMPLES as f32), -0.1f32..5f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        (0..data.len()).map(|i| (i as f32, data[i])),
        &BLACK,
    ))?;

    chart.draw_series(LineSeries::new(
        (0..=1).map(|x| ((N_SAMPLES as f32) * (x as f32), param)),
        &BLACK,
    ))?;

    let mut idx: Vec<_> = (2..=N_SAMPLES).map(|x| x as f32).collect();
    chart.draw_series(LineSeries::new(
        (0..est.len()).map(|i| (idx[i], est[i].0)),
        &RED,
    ))?;

    let mut interval: Vec<_> = (0..idx.len()).map(|i| est[i].0 + cnf[i]).collect();
    interval.extend((0..idx.len()).rev().map(|i| est[i].0 - cnf[i]));
    idx.extend(idx.clone().iter().rev());

    let vertices: Vec<_> = (0..interval.len()).map(|i| (idx[i], interval[i])).collect();
    chart.draw_series(std::iter::once(Polygon::new(vertices, RED.mix(0.2))))?;

    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}
