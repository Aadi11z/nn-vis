# nn-vis
A simple Neural Network Visualiser
# rust notes
|x| expr is a closure (like an inline function) that takes a parameter x.
|_| expr is the same, but the _ means “I don’t care about this parameter, I’m not going to use it”.
.map(|_| ...) is looping over 0..output_size. Each iteration ignores the actual number from the range (0, 1, 2, …), because you only care about repeating the same process output_size times.
The inner .map(|_| rng.gen_range(-1.0..1.0)) is the same: it just says “generate a random number for each input connection, don’t care about the loop index.”
So |_| is just a way of saying: “I need to repeat something N times, but I don’t care about the loop variable.”

## when passing arguments to a function:
- fn A(x: i32) => x here is a copy but still usable elsewhere
- fn A(x: String) => x here is not a copy, the text is moved to the funciton and isnt usable elsewhere
- fn A(x: &String) => x here is being borrowed immutably, and is usable elsewhere
- fn A(x: &mut String) => x here is being mutably borrowed, only one mutable borrow at a time, modified in place
- fn A(x: i32) -> i32 => returns an i32 value back to caller
- fn A<T: std::fmt::Debug>(val: T) => implements the Debug trait (need to learn more abt Traits!!)
- fn A(x: &[i32]) -> i32 => Pass a refernce to a slice (slice is a reference to an array/vector/string, provides a view w/o ownership)
- fn A(&self) => &self means borrow immutably, just read info
- fn A(&mut self) => &mut self means borrow mutably, change info
- fn A(self) => self's ownership is taken/consumed
- fn A(neurons: usize) -> Self { Self { neurons } } => A here acts as a constructor, Returns the stuct type of the impl the fn A is defined in
