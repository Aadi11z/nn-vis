use std::io;
mod network;
use network::NeuralNetwork;

fn read_input<T: std::str::FromStr>(msg: &str) -> T{
    loop {
        let mut input = String::new();
        println!("{}", msg);
        io::stdin()
            .read_line(&mut input)
            .expect("Couldn't read line, check inputting function.");
        match input.trim().parse::<T>() {
            Ok(val) => return val,
            Err(_) => println!("Invalid input, try again."),
        }
    }
}
fn main() {
    let mut network = NeuralNetwork::new(&[2, 3, 1]);
    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    let learning_rate = 0.1;
    for epoch in 0..5000 {
        let mut loss = 0.0;
        
        for (input, target) in &data {
            let prediction = network.forward(input);
            loss += mean_squared_error(&prediction, target);
            network.backward(input, target, learning_rate);
        }
        
        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss / data.len() as f64);
        }
    }
}