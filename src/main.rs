mod network;
use network::InputLayer;
use network::Layer;

fn main() {
    let _learning_rate = 0.1;
    let _y_target = 1.00;
    let mut inputs= InputLayer::new(vec![0.0, 1.0]);
    let weights = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
    let biases = vec![1.0, 2.0];
    let mut hidden = Layer::new(weights, biases);
    let hidden_outputs = hidden.forward_prop(&mut inputs);

    println!("{:?}", hidden_outputs);
}

// fn read_input<T: std::str::FromStr>(msg: &str) -> T{
//     loop {
//         let mut input = String::new();
//         println!("{}", msg);
//         io::stdin()
//             .read_line(&mut input)
//             .expect("Should never fail to read input");
//         match input.trim().parse::<T>() {
//             Ok(val) => return val,
//             Err(e) => println!("Error: Invalid Input"),
//         }
//     }
// }