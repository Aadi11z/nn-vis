//                                    b1 
// Input (0) -> w1 \                  |
//                  -> Hidden Neuron (h1) -> w5    
// Input (1) -> w3 /                            \
//                                               --> Output Neuron (y) Error = y^ - y
// Input (0) -> w2 \                            /                   |
//                  -> Hidden Neuron (h2) -> w6                     b3
// Input (1) -> w4 /                  |                             
//                                    b2                            
use rand::Rng;

fn sigmoid(x: f64) -> f64{
    1.0/(1.0 + (-x).exp())
}
pub struct InputLayer{
    pub inputs: Vec<f64>,
}

impl InputLayer{
    fn new() -> Self{ 
        InputLayer { inputs: Vec::new() }
    }
}

pub struct Layer{
    pub weights: Vec<Vec<f64>>, // [1, 2, 3, 4, 5, 6]
    pub biases: Vec<f64>, // [1, 2, 3]
}

impl Layer{
    pub fn new() -> Self{
        Layer { weights: Vec::new(), biases: Vec::new() }
    }

    pub fn forward_prop(&mut self, neuron_inputs: &mut InputLayer) -> Vec<f64> {
        let mut outputs = Vec::new();

        for (neuron_weights, neuron_bias) in self.weights.iter().zip(self.biases.iter()) {
            let h: f64 = neuron_weights.iter()
                .zip(neuron_inputs.inputs.iter())
                .map(|(w, i)| w * i)
                .sum::<f64>() + neuron_bias;
            outputs.push(sigmoid(h));
        }
        outputs
    }

    

    pub fn backward_prop(&mut self, inputs: &mut InputLayer) {

    }
}
