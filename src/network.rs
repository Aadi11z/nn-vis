//                                    b1
// Input (0) -> w1 \                  |
//                  -> Hidden Neuron (h1) -> w5
// Input (1) -> w3 /                            \
//                                               --> Output Neuron (y) Error = y^ - y
// Input (0) -> w2 \                            /                   |
//                  -> Hidden Neuron (h2) -> w6                     b3
// Input (1) -> w4 /                  |
//                                    b2
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
#[derive(Debug)]
pub struct InputLayer {
    pub inputs: Vec<f64>,
}

impl InputLayer {
    pub fn new(inputs: Vec<f64>) -> Self {
        InputLayer { inputs }
    }
}

pub enum WeightsType {
    Single(Vec<f64>),
    Multiple(Vec<Vec<f64>>),
}
pub struct Layer {
    pub weights: WeightsType, // [1, 2, 3, 4, 5, 6]
    pub biases: Vec<f64>,      // [1, 2, 3]
}

impl Layer {
    pub fn new(weights: WeightsType, biases: Vec<f64>) -> Self {
        Layer { weights, biases }
    }

    pub fn forward_prop(&mut self, neuron_inputs: &InputLayer) -> Vec<f64> {
        let mut outputs = Vec::new();

        match &self.weights {
            WeightsType::Single(weights_vec) => {
                let h = weights_vec
                    .iter()
                    .zip(neuron_inputs.inputs.iter())
                    .map(|(w, i)| w * i)
                    .sum::<f64>()
                    +self.biases[0];
                outputs.push(sigmoid(h));
            }
            WeightsType::Multiple(weights_matrix) => {
                for (neuron_weights, neuron_bias) in weights_matrix.iter().zip(self.biases.iter()) {
                    let h: f64 = neuron_weights
                        .iter()
                        .zip(neuron_inputs.inputs.iter())
                        .map(|(w, i)| w * i)
                        .sum::<f64>()
                        + neuron_bias;
                    outputs.push(sigmoid(h));
                }
            }
        }
        outputs
    }
}
