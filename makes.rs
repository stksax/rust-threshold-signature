struct Input {
    key_share : u128,
    input_r : u128,
    output_max : u128,
    output_min : u128,
}

impl Input{
    fn output_key_share(&self) -> Vec<u128>{
        let mut output_key_share = vec![self.key_share; self.output_max];

        for i in self.output_max{
            for j in self.output_min-1{
                //p(i) = u + ia + (ia)**2 + ...
                output_key_share[i] += ((i+1) * self.input_r)**(j+1);
            }
        }
        output_key_share
    }
}



fn main() {
    let input = Input {
        output_max: 5,
        key_share: 123,
        output_num: 3,
        input_r: 2,
    };

    let result = input.output_key_share();
    println!("{:?}", result);
}
