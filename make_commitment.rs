use std::ops::Mul;
use pasta_curves::{group::{cofactor::CofactorCurveAffine, ff::PrimeField, Curve}, pallas};

use crate::*;
use myp::{Encrypt,EncryptAddMut,Decrypt};

//send cipher k
struct FirstStep{
    selfk : Vec<u128>,
    mta_pub_n : Vec<u128>,
}

impl FirstStep {
    fn encrypt_k(&self) -> Vec<u128>{
        let mut random_r = Vec::new();
        let mut iter = 0;
        for i in self.mta_pub_n.clone(){
            random_r.push(generate_random_u128_in_range(1, i));
            if i > std::u64::MAX as u128{
                panic!("pub_n should < u64")
            }
        }

        let mut encrypt_instance = Vec::new();
        for (i, j) in self.selfk.iter().zip(&self.mta_pub_n) {
            if i > j{
                panic!("pub_n should < u64")
            }
            encrypt_instance.push(Encrypt {
                mta_pub_n: *j,
                rand: random_r[iter],
                message: *i,
            });
            iter+=1;
        }

        let mut cipher_k = Vec::new();
        for i in encrypt_instance {
            cipher_k.push(i.encrypt());
        }

        cipher_k 
    }
}

//recive cipherk and generate cipher(k2*r1+c1) and keep -c1 as key
struct SecondStep{
    others_mta_pub_n : Vec<u128>,
    others_cipher_k : Vec<u128>,
    selfr : Vec<u128>,
    selfk : Vec<u128>,
}

impl SecondStep {
    fn cipher_k(&self) -> (Vec<u128>, Vec<u128>){
        // let random_unm = generate_random_u128_in_range(1,  self.mta_pub_n);
        let mut add_num_neg = Vec::new();
        let mut random_num2 = Vec::new();
        for _ in self.others_mta_pub_n.iter(){
            for ((i, j), k) in self.others_mta_pub_n.iter().zip(&self.selfr).zip(&self.selfk){
                add_num_neg.push(generate_random_u128_in_range(1, *j * *k));
                random_num2.push(generate_random_u128_in_range(1, *i));
            }
        }
        
        let mut iter : usize = 0;
        let mut encrypt_instance = Vec::new();
        for k in &self.selfr {
            for (i, j) in self.others_mta_pub_n.iter().zip(&self.others_cipher_k) {
                encrypt_instance.push(EncryptAddMut {
                    mta_pub_n : *i,
                    cipher : *j,
                    add_num : add_num_neg[iter],
                    mut_num : *k,
                    rand : random_num2[iter],
                });
                iter+=1;
            }
        }
        
        let mut cipher_k2w1_plus_rand = Vec::new();
     
        for i in encrypt_instance{
            cipher_k2w1_plus_rand.push(i.mut_and_add());
        }
        
        (cipher_k2w1_plus_rand,  add_num_neg)
    }
}

//decrypt cipher get k1*r2+c2
struct ThirdStep{
    input_p : Vec<u128>,
    input_q : Vec<u128>,
    cipher : Vec<u128>,
}

impl ThirdStep {
    fn decrypt_cipher(&self) -> Vec<u128>{
        let mut pri_keyp = Vec::new();
        let mut pri_keyq = Vec::new();
        for (p, q) in self.input_p.iter().zip(&self.input_q){
            pri_keyp.push(*p);
            pri_keyq.push(*q);
        }

        let mut iter= 0;
        let mut encrypt_instance = Vec::new();
        
        for i in &self.cipher {
            encrypt_instance.push( Decrypt{
                pri_p : pri_keyp[iter],
                pri_q : pri_keyq[iter],
                cipher : *i,
            });
            iter+=1;
            if iter == pri_keyp.len(){
                iter=0;
            }
        }

        let mut plain_text = Vec::new();
        for i in encrypt_instance{
            plain_text.push(i.decrypt());
        }
      
        plain_text
    }
}

struct FourthStep{
    plain_text : Vec<u128>,
    selfk : Vec<u128>,
    selfr : Vec<u128>,
    add_num_neg : Vec<u128>,
}

impl FourthStep {
    fn combine(&self) -> (Vec<pallas::Scalar>, Vec<pallas::Affine>){
        let mut sharding_commitment = Vec::new();
        //k1r1
        let mut ifq = pallas::Scalar::one();
        let mut jfq = pallas::Scalar::one();
        for (i ,j) in self.selfk.iter().zip(&self.selfr){
            ifq = pallas::Scalar::from_u128(*i);
            jfq = pallas::Scalar::from_u128(*j);
            sharding_commitment.push(pallas::Scalar::mul(&ifq, &jfq));
        }
        
        //k1r1 + plentext - add_num_neg = (k1 + k2 + ....) * (w1 +w2 + ....)
        let mut player = 0;
        let mut raw = 0;
        for i in &self.plain_text{
            ifq = pallas::Scalar::from_u128(*i);
            if player != raw{
                sharding_commitment[player] = pallas::Scalar::add(&sharding_commitment[player], &ifq);
            }
            player+=1;
            if player == self.selfk.len(){
                player=0;
                raw+=1;
            }
        }

        player = 0;
        raw = 0;
        for j in &self.add_num_neg{
            jfq = pallas::Scalar::from_u128(*j);
            if player != raw{
                sharding_commitment[raw] = pallas::Scalar::sub(&sharding_commitment[raw], &jfq);
            }
            player+=1;
            if player == self.selfk.len(){
                player=0;
                raw+=1;
            }
        }
        
        let mut verify_point = Vec::new();
        let affine_generator = pallas::Affine::generator();
        #[allow(unused_assignments)]
        let mut selfr_fq = pallas::Scalar::zero();
        for i in &self.selfr{
            selfr_fq = pallas::Scalar::from_u128(*i);
            verify_point.push(pallas::Affine::mul(affine_generator, selfr_fq).to_affine());
        }
    
        (sharding_commitment, verify_point)
    }
}

pub fn make_commitment(
    selfk_vec : Vec<u128>,
    selfr_vec : Vec<u128>,
    mta_pub_n_vec : Vec<u128>,
    mta_pri_p_vec : Vec<u128>,
    mta_pri_q_vec : Vec<u128>,
) -> (Vec<pallas::Scalar>, Vec<pasta_curves::EpAffine>){
    //k * r should < n , because it will mod n
    for ((i, j), k) in selfk_vec.iter().zip(&selfr_vec).zip(&mta_pub_n_vec){
        if *i * * j >= *k{
            panic!("k * r should < n , because it will mod n");
        }
    }

    let step_1 = FirstStep{
        selfk : selfk_vec.clone(),
        mta_pub_n : mta_pub_n_vec.clone(),
    };
    let cipher_k = step_1.encrypt_k();

    let step_2 = SecondStep{
        others_mta_pub_n : mta_pub_n_vec,
        others_cipher_k : cipher_k,
        selfr : selfr_vec.clone(),
        selfk : selfk_vec.clone(),
    };
    let (cipher_k2r1_plus_c1, neg_num) = step_2.cipher_k();
 
    let step_3 = ThirdStep{
        input_p : mta_pri_p_vec,
        input_q : mta_pri_q_vec,
        cipher : cipher_k2r1_plus_c1,
    };
    let plain_text = step_3.decrypt_cipher();

    let step_4 = FourthStep{
        plain_text : plain_text,
        selfk : selfk_vec,
        selfr : selfr_vec,
        add_num_neg : neg_num,
    };
    let (sharding_commitment, verify_point) = step_4.combine();

    (sharding_commitment, verify_point)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn text(){
    //k * r should < n , because it will mod n
    let allice_selfk = 564;
    let allice_selfr = 345;
    let allice_mta_pri_p = 37057;
    let allice_mta_pri_q = 55021;
    let allice_mta_pub_n = allice_mta_pri_p * allice_mta_pri_q;
    if (allice_selfk * allice_selfr) >= allice_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((allice_selfk*allice_selfr) < allice_mta_pub_n);

    let bob_selfk = 687;
    let bob_selfr = 466;
    let bob_mta_pri_p = 45497;
    let bob_mta_pri_q = 61363;
    let bob_mta_pub_n = bob_mta_pri_p * bob_mta_pri_q;
    if (bob_selfk * bob_selfr) >= bob_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((bob_selfk*bob_selfr) < bob_mta_pub_n);

    let chris_selfk = 745;
    let chris_selfr = 531;
    let chris_mta_pri_p = 58237;
    let chris_mta_pri_q = 50129;
    let chris_mta_pub_n = chris_mta_pri_p * chris_mta_pri_q;
    if (chris_selfk * chris_selfr) >= chris_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((chris_selfk*chris_selfr) < chris_mta_pub_n);

    let selfk_vec = [allice_selfk, bob_selfk, chris_selfk].to_vec();
    let selfr_vec = [allice_selfr, bob_selfr, chris_selfr].to_vec();
    let mta_pub_n_vec = [allice_mta_pub_n, bob_mta_pub_n, chris_mta_pub_n].to_vec();
    let mta_pri_p_vec = [allice_mta_pri_p, bob_mta_pri_p, chris_mta_pri_p].to_vec();
    let mta_pri_q_vec = [allice_mta_pri_q, bob_mta_pri_q, chris_mta_pri_q].to_vec();

    let (sharding_commitment, _) = make_commitment(
        selfk_vec,
        selfr_vec,
        mta_pub_n_vec,
        mta_pri_p_vec,
        mta_pri_q_vec
    );
    

    let v1 = allice_selfk + bob_selfk + chris_selfk;
    let v2 = allice_selfr + bob_selfr + chris_selfr;
    let v3 = v1 * v2;
    let mut v4 = pallas::Scalar::zero();
    for i in &sharding_commitment{
        v4 = pallas::Scalar::add(&v4, i);
    }
    let v5 = pallas::Scalar::from_u128(v3);
    assert_eq!(v5, v4)
}
}
