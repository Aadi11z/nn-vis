fn main(){
    let mut v = vec![1.0, 2.0, 3.0];
    // Immutable iteration
    for x in v.iter() {
        println!("{}", x); // x is &f32
    }

    // Mutable iteration
    for x in v.iter_mut() {
        *x += 1.0; // change value in place
    }

    // Consume vector and iterate
    for x in v.into_iter() {
        println!("{}", x); // x is f32, ownership moved
    }
}