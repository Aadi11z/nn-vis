use rand::Rng;

fn sigmoid(x: f32) -> f32{
    1.0/(1.0 + (-x).exp())
}
pub struct Neuron{
    pub weights: Vec<f32>,
    pub bias: f32,
    pub output: f32
}

impl Neuron{
    pub fn new(input_size: u32) -> Self{
        let mut rng = rand::rng();
        let weights: Vec<f32> = (0..input_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let bias = rng.random_range(-1.0..1.0);

        Self { weights, bias, output: 0.0 }
    }
    pub fn forward(&mut self, inputs: &[f32]) -> f32 {
        let sum: f32 = self.weights.iter()
            .zip(inputs)
            .map(|(w, i)| w * i)
            .sum::<f32>() + self.bias;

        self.output = sigmoid(sum);
        self.output
    }
}