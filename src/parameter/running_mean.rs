pub struct RunningMean {
    mean: f32,
    m2: f32,
    cnt: usize,
}

impl RunningMean {
    pub fn new() -> Self {
        RunningMean {
            mean: 0.0,
            m2: 0.0,
            cnt: 0,
        }
    }

    pub fn update(&mut self, new_val: f32) {
        self.cnt += 1;

        let delta = new_val - self.mean;
        self.mean += delta / (self.cnt as f32);

        let delta2 = new_val - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn get(&self) -> Result<(f32, f32), &'static str> {
        match self.cnt {
            0 => Err("No measurements recieved"),
            1 => Err("No measurements recieved"),
            _ => Ok((self.mean, self.m2 / ((self.cnt as f32) - 1.0))),
        }
    }

    pub fn confidence(&self, percentile: f32) -> Result<f32, &'static str> {
        match self.cnt {
            0 | 1 => Err("Need at least two measuerments for confidence interval"),
            _ => Ok((percentile * (self.m2 / ((self.cnt as f32) - 1.0)).sqrt())
                / (self.cnt as f32).sqrt()),
        }
    }
}
