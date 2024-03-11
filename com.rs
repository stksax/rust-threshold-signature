use std::{ops::Mul, process::Termination};

use halo2_proofs::arithmetic::Field;
use pasta_curves::{group::{cofactor::CofactorCurveAffine, ff::PrimeField, Curve}, pallas};
mod mta; 
use mta::*;
use rand::Rng;
pub fn generate_random_u128_in_range(min: u128, max: u128) -> u128 {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(min..=max);
    random_value
}
//send cipher k
struct FirstStep{
    selfk : u128,
    mta_pub_n : u128,
}

impl FirstStep {
    fn cipher_k(&self) -> u128{
        let random_r = generate_random_u128_in_range(1, self.mta_pub_n);
        let encrypt_instance = Encrypt {
            mta_pub_n: self.mta_pub_n,
            input_r: random_r,
            message: self.selfk, 
        };  

        let cipher_k = encrypt_instance.encrypt();
        cipher_k 
    }
}

//recive cipherk and generate cipher(k2*r1+c1) and keep -c1 as key
struct SecondStep{
    others_mta_pub_n : u128,
    others_cipher_k : u128,
    selfr : u128,
}

impl SecondStep {
    fn cipher_k(&self) -> (u128, u128){
        // let random_unm = generate_random_u128_in_range(1,  self.mta_pub_n);
        let random_unm: u128 = generate_random_u128_in_range(1,  1234);
        let random_unm2 = generate_random_u128_in_range(1,  self.others_mta_pub_n);
        let encrypt_instance = EncryptAddMut {
            mta_pub_n : self.others_mta_pub_n,
            cipher : self.others_cipher_k,
            add_num : random_unm,
            mut_num : self.selfr,
            rand : random_unm2,
        };
        let cipher_k2r1 = encrypt_instance.encrypt_mut();

        let encrypt_instance2 = EncryptAddMut {
            mta_pub_n : self.others_mta_pub_n,
            cipher : cipher_k2r1,
            add_num : random_unm,
            mut_num : self.selfr,
            rand : random_unm2,
        };
        let cipher_k2r1_plus_c = encrypt_instance2.encrypt_add();
        let add_num_neg = random_unm;
        (cipher_k2r1_plus_c,  add_num_neg)
    }
}

//decrypt cipher get k1*r2+c2
struct ThirdStep{
    input_p : u128,
    input_q : u128,
    mta_pub_n : u128,
    cipher : u128,
}

impl ThirdStep {
    fn decrypt_cipher(&self) -> u128{
        let encrypt_instance = Decrypt {
            input_p : self.input_p,
            input_q : self.input_q,
            mta_pub_n : self.mta_pub_n,
            cipher : self.cipher,
        };

        let plain_text = encrypt_instance.decrypt();
        plain_text
    }
}

struct FourthStep{
    plain_text : u128,
    selfk : u128,
    selfr : u128,
    add_num_neg : u128,
}

impl FourthStep {
    fn combine(&self) -> (u128, pallas::Affine){
        // let selfk = pallas::Scalar::from_u128(self.selfk);
        // let selfr = pallas::Scalar::from_u128(self.selfr);
        // let k1r1 = pallas::Scalar::mul(&selfk, &selfr);
        // let k1r2 = pallas::Scalar::from_u128(self.plain_text);
        // let k2r1 = self.add_num_neg;
        // let add1 = pallas::Scalar::add(&k1r1, &k1r2);
        // let sharding_commitment = pallas::Scalar::add(&add1, &k2r1);
        // let sharding_commitment = pallas::Scalar::invert(&add2).unwrap();
        let k1r1 = self.selfk * self.selfr;
        let sharding_commitment = k1r1 + self.plain_text - self.add_num_neg;

        let affine_generator = pallas::Affine::generator();
        let selfr = pallas::Scalar::from_u128(self.selfr);
        let verify_point_ep = pallas::Affine::mul(affine_generator, &selfr);
        let verify_point = pallas::Point::to_affine(&verify_point_ep);

        (sharding_commitment, verify_point)
    }
}

fn main() {
    //k * r should < n , because it will mod n
    let allice_selfk = 564;
    let allice_selfr = 345;
    let allice_mta_pri_p = 3163;
    let allice_mta_pri_q = 3541;
    let allice_mta_pub_n = allice_mta_pri_p * allice_mta_pri_q;
    if (allice_selfk * allice_selfr) >= allice_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((allice_selfk*allice_selfr) < allice_mta_pub_n);

    let bob_selfk = 687;
    let bob_selfr = 466;
    let bob_mta_pri_p = 3347;
    let bob_mta_pri_q = 2939;
    let bob_mta_pub_n = bob_mta_pri_p * bob_mta_pri_q;
    if (bob_selfk * bob_selfr) >= bob_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((bob_selfk*bob_selfr) < bob_mta_pub_n);


    let a_step_1 = FirstStep{
        selfk : allice_selfk,
        mta_pub_n : allice_mta_pub_n,
    };
    let cipher_a1 = a_step_1.cipher_k();

    let b_step_1 = FirstStep{
        selfk : bob_selfk,
        mta_pub_n : bob_mta_pub_n,
    };
    let cipher_b1 = b_step_1.cipher_k();

    let a_step_2 = SecondStep{
        others_mta_pub_n : bob_mta_pub_n,
        others_cipher_k : cipher_b1,
        selfr : allice_selfr,
    };
    let (cipher_k2r1_plus_c1, neg_c1) = a_step_2.cipher_k();

    let b_step_2 = SecondStep{
        others_mta_pub_n : allice_mta_pub_n,
        others_cipher_k : cipher_a1,
        selfr : bob_selfr,
    };
    let (cipher_k1r2_plus_c2, neg_c2) = b_step_2.cipher_k();

    let a_step_3 = ThirdStep{
        input_p : allice_mta_pri_p,
        input_q : allice_mta_pri_q,
        mta_pub_n : allice_mta_pub_n,
        cipher : cipher_k1r2_plus_c2,
    };
    let plain_text_a = a_step_3.decrypt_cipher();

    let b_step_3 = ThirdStep{
        input_p : bob_mta_pri_p,
        input_q : bob_mta_pri_q,
        mta_pub_n : bob_mta_pub_n,
        cipher : cipher_k2r1_plus_c1,
    };
    let plain_text_b = b_step_3.decrypt_cipher();

    let a_step_4 = FourthStep{
        plain_text : plain_text_a,
        selfk : allice_selfk,
        selfr : allice_selfr,
        add_num_neg : neg_c1,
    };
    let (sharding_commitment_a, verify_point_a) = a_step_4.combine();

    let b_step_4 = FourthStep{
        plain_text : plain_text_b,
        selfk : bob_selfk,
        selfr : bob_selfr,
        add_num_neg : neg_c2,
    };
    let (sharding_commitment_b, verify_point_b) = b_step_4.combine();

    let v1 = allice_selfk + bob_selfk;
    let v2 = allice_selfr + bob_selfr;
    let v3 = v1 * v2;
    let v4 = sharding_commitment_b + sharding_commitment_a;
    println!("{:?}",&v3);
    println!("{:?}",&v4);
}