use std::ops::{Add, Mul};
use halo2_proofs::arithmetic::Field;
use rand::Rng;

use pasta_curves::{group::{cofactor::CofactorCurveAffine, ff::PrimeField, Curve}, pallas};
//p(i) = u + ir + (ir)**2 + ...
fn generate_random_u128_in_range(min: u128, max: u128) -> u128 {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(min..=max);
    random_value
}

pub struct Input {
    pub key_share : u128,
    pub rand_num : u128,
    pub output_max : usize,
    pub output_min : usize,
}

impl Input{
    pub fn output_key_share(&self) -> Vec<u128>{
        let mut output_key_share = vec![self.key_share; self.output_max];

        for i in 0..self.output_max{
            for j in 0..(self.output_min-1){
                output_key_share[i] += ((i+1) as u128* self.rand_num).pow((j as u32)+1);
            }
        }
        output_key_share
    }
}

pub struct CollectOutputKeyShare {
    pub key_share : Vec<u128>,
    pub member : u128,
    pub self_num : u128,
}

//member is all the player join the signature
impl CollectOutputKeyShare {
    pub fn collect(&self) -> (u128, pallas::Affine) {
        let mut self_key_share :u128 = 0;
        let member = self.member;
        let mut count =0;
        for i in &self.key_share{
            count += 1;
            if count == self.self_num || ((count+member-self.self_num))%member==0{
                self_key_share += i;
            }
            
        }
        let key_fq = pallas::Scalar::from_u128(self_key_share);
        let generator = pallas::Affine::generator();
        let result = pallas::Affine::mul(generator, key_fq).to_affine();

        (self_key_share, result)
    }
}

pub struct CalculatePubKey {
    pub degree : u128,
    pub coefficient : Vec<u128>,
    pub pub_key : Vec<pallas::Affine>,
}

impl CalculatePubKey {
    pub fn calculate(&self) -> pallas::Affine{
        let thousandfq = pallas::Scalar::from_u128(1000);
        let mut num = 0;
        let mut result = pallas::Affine::identity();
        while num < self.degree {
            let index: usize = num.try_into().unwrap();
            let mut iter = 0;
            let mut ans = self.pub_key[index];
            for i in self.coefficient.iter(){                
                if iter != num {
                    let self_coefficient :u128 = self.coefficient[index];
                    let mut dev: u128 = 1000;
                    let mutn = pallas::Scalar::from_u128(i.clone());
                    ans = pallas::Affine::mul(ans, &mutn).to_affine();
                    dev  =  dev + i - self_coefficient;
                    let dev2 = pallas::Scalar::from_u128(dev);
                    let dev3 = pallas::Scalar::sub(&dev2, &thousandfq);
                    let dev4 = pallas::Scalar::invert(&dev3).unwrap();
                    ans = pallas::Affine::mul(ans, dev4).to_affine();
                }
                iter+=1;
            }
            result = pallas::Affine::add(result, ans).to_affine();
            num+=1;
        }   
        result
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn key_generate_test() {
        let key_share1 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let key_share2 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let key_share3 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let key_share4 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let key_share5 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let input1 = Input{
            key_share : key_share1,
            rand_num : 379278,
            output_max : 5,
            output_min : 3,
        };
        let result1 = input1.output_key_share();
    
        let input2 = Input{
            key_share : key_share2,
            rand_num : 4812738974,
            output_max : 5,
            output_min : 3,
        };
        let result2 = input2.output_key_share();
    
        let input3 = Input{
            key_share : key_share3,
            rand_num : 43217,
            output_max : 5,
            output_min : 3,
        };
        let result3 = input3.output_key_share();
    
        let input4 = Input{
            key_share : key_share4,
            rand_num : 745,
            output_max : 5,
            output_min : 3,
        };
        let result4 = input4.output_key_share();
    
        let input5 = Input{
            key_share : key_share5,
            rand_num : 542,
            output_max : 5,
            output_min : 3,
        };
        let result5 = input5.output_key_share();
    
        let mut user_vec = Vec::new();
        user_vec.extend(result1);
        user_vec.extend(result2);
        user_vec.extend(result3);
        user_vec.extend(result4);
        user_vec.extend(result5);

        let user1 = CollectOutputKeyShare{
            key_share : user_vec.clone(),
            member : 5,
            self_num : 1,
        };
        let (user1_prikey_share, user1_pubket_share) = user1.collect();
        
        let user2 = CollectOutputKeyShare{
            key_share : user_vec.clone(),
            member : 5,
            self_num : 2,
        };
        let (user2_prikey_share, user2_pubket_share) = user2.collect();
    
        let user3 = CollectOutputKeyShare{
            key_share : user_vec.clone(),
            member : 5,
            self_num : 3,
        };
        let (user3_prikey_share, user3_pubket_share) = user3.collect();
    
        let pub_key = CalculatePubKey {
            degree : 3,
            coefficient : [1,2,3].to_vec(),
            pub_key : [user1_pubket_share, user2_pubket_share, user3_pubket_share].to_vec(),
        };
        let result = pub_key.calculate();
        let prik = key_share1 + key_share2 + key_share3 + key_share4 + key_share5;
        let check = pallas::Scalar::from_u128(prik);
        let generator = pallas::Affine::generator();
        let check2 = pallas::Affine::mul(generator, check).to_affine();
        let ans = pallas::Affine::eq(&result, &check2);
  
        assert_eq!(ans, true)
    }
}