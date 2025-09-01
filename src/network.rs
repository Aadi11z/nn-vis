use rand::Rng;

fn sigmoid(x: f32) -> f32{
    1.0/(1.0 + (-x).exp())
}

pub fn mean_squared_error(predicted: &[f64], actual: &[f64]) -> f64 {
    predicted.iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>() / predicted.len() as f64
}

pub struct Layer{
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>
}

impl Layer{
    fn new(input_size: usize, output_size: usize) -> Layer{
        let mut rng = rand::rng();

        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        
        let biases = (0..output_size).map(|_| rng.random_range(-1.0..1.0)).collect();
        
        Layer { weights, biases }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32>{
        self.weights.iter().enumerate().map(|(i, neuron_weights)| {
            let sum: f64 = neuron_weights.iter().zip(input.iter())
                .map(|(w, i)| w * i)
                .sum();
            sigmoid(sum + self.biases[i])
        }).collect()
    }

    fn backward(&mut self, input: &[f64], error: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut input_error = vec![0.0; input.len()];
        
        for (i, neuron_weights) in self.weights.iter_mut().enumerate() {
            for (j, weight) in neuron_weights.iter_mut().enumerate() {
                input_error[j] += *weight * error[i];
                *weight -= learning_rate * error[i] * input[j];
            }
            self.biases[i] -= learning_rate * error[i];
        }
        
        input_error
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> NeuralNetwork {
        let layers = layer_sizes.windows(2)
            .map(|w| Layer::new(w[0], w[1]))
            .collect();
        NeuralNetwork { layers }
    }
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.layers.iter().fold(input.to_vec(), |acc, layer| layer.forward(&acc))
    }

    pub fn backward(&mut self, inputs: &[f64], target: &[f64], learning_rate: f64) {
        // Perform a forward pass and store intermediate layer inputs
        let mut layer_inputs = vec![inputs.to_vec()]; // Store inputs to each layer
        let mut current_input = inputs.to_vec();
        
        for layer in &self.layers {
            current_input = layer.forward(&current_input);
            layer_inputs.push(current_input.clone());
        }
        
        // Calculate initial error
        let error = layer_inputs.last().unwrap() // Output of the last layer
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect::<Vec<_>>();
        
        let mut current_error = error;
        
        // Backward pass
        for (layer, inputs) in self.layers.iter_mut().rev().zip(layer_inputs.iter().rev().skip(1)) {
            current_error = layer.backward(inputs, &current_error, learning_rate);
        }
    }
}
