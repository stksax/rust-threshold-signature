use primal::Primes;
use rand::seq::SliceRandom;
use rand::Rng;
// pub fn rand_prime() -> u128{
//     let mut rng = rand::thread_rng();
//     let primes = Primes::all().skip(1000).take(400000).collect::<Vec<_>>(); 
//     let prime = *primes.choose(&mut rng).unwrap() as u128; 
//     let max = std::u64::MAX;
//     println!("max{}",max);
//     let last = primes.last().unwrap();
//     println!("last{}",last);
//     prime
// }
pub fn generate_random_u128_in_range(min: u128, max: u128) -> u128 {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(min..=max);
    random_value
}
//use 31 because i32 can only <<31
pub fn rand_prime() -> u128{
    let rand = generate_random_u128_in_range(1, 15);
    let a = (1<<16) as u128;
    let b = (1<<rand) as u128;
    let prime = a - b + 1;
    prime
}