// main.rs
mod network;
use network::{InputLayer, Layer, WeightsType};

fn main() {
    let _learning_rate = 0.1;
    let _y_target = 1.00;

    // Inputs
    let inputs = InputLayer::new(vec![0.0, 1.0]);

    // Input -> Hidden
    let weights_1 = WeightsType::Multiple(vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    let biases_1 = vec![1.0, 2.0];
    let mut input_to_hidden = Layer::new(weights_1, biases_1);
    let hidden_outputs = InputLayer::new(input_to_hidden.forward_prop(&inputs));

    println!("Hidden layer outputs: {:?}", hidden_outputs);

    // Hidden -> Output
    let weights_2 = WeightsType::Single(vec![5.0, 6.0]);
    // let weights_2 = WeightsType::Multiple(vec![vec![5.0], vec![6.0]]);
    let biases_2 = vec![3.0];
    let mut hidden_to_output = Layer::new(weights_2, biases_2);
    let y = hidden_to_output.forward_prop(&hidden_outputs);

    println!("Final output: {:?}", y);
    let error = _y_target - y[0];
    println!("Error: {}", error);

}