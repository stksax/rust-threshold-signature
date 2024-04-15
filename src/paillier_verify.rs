use std::ops::{Add, Mul};

use ff::PrimeField;
use pasta_curves::{group::{cofactor::CofactorCurveAffine, Curve}, pallas};
use crate::generate_random_u128_in_range;

pub fn pow_and_mod(mut base: u128, mut exponent: u128, modulus: u128) -> u128{
    if exponent == 0 {
        return 1;
    }

    while exponent & 1 == 0 {
        base = base.clone() * base;
        base = base % modulus;
        exponent >>= 1;
    }
    if exponent == 1 {
        base = base % modulus;
    }

    let mut acc = base.clone();

    while exponent > 1 {
        exponent >>= 1;
        base = base.clone() * base;
        base = base % modulus;
        if exponent & 1 == 1 {
            acc = acc * base.clone();
            acc = acc % modulus;
        }
    }
    acc
}

pub fn find_mod_inverse(x: u128, modulus: u128) -> u128 {
    let mut k = 1;
    let mut k2 = 0;
    let mut ans = 0;

    while k < modulus {
        k2 = (k * modulus)+1;
        if k2 % x == 0{
            ans = k2 / x;
            break;
        }
        k+=1;
    }
    ans
}

pub struct PrepareZk{
    n : u128,
    secret : u128,
    r : u128,
}

impl PrepareZk{
    pub fn prepare_zk_verify(&self) -> ([pallas::Affine;2],[u128;9]){
        let n = self.n;
        let tau = n+1;
        let secret = self.secret;
        let generator = pallas::Affine::generator();
        let y = pallas::Affine::mul(generator, pallas::Scalar::from_u128(secret)).to_affine();
        let range_q = generate_random_u128_in_range(1, std::u16::MAX as u128);

        let h1 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let h2 = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let a = generate_random_u128_in_range(1, range_q);
        let b = generate_random_u128_in_range(1, n);
        let p = generate_random_u128_in_range(1, range_q*n);
        let r = self.r;

        let z1 = pow_and_mod(h1, secret, n);
        let z2 = pow_and_mod(h2, p, n);
        let z = (z1 * z2) % n;
        let u1 = pallas::Affine::mul(generator, pallas::Scalar::from_u128(a)).to_affine();
        let u21  = pow_and_mod(tau, a, n*n);
        let u22  = pow_and_mod(b, n, n*n);
        let u2 = (u21 * u22) % (n*n);
        let u31 = pow_and_mod(h1, a, n);
        let u32 = pow_and_mod(h2, r, n);
        let u3 = (u31 * u32) % n;
        let e = generate_random_u128_in_range(1, std::u8::MAX as u128);
        let s1 = (e * secret) + a;
        let s20 = pow_and_mod(r, e, n*n);
        let s2 = s20 * b;
        let s3 = (e * p) + r;

        (
            [u1,y],
            [z,u2,u3,e,s1,s2,s3,h1,h2],
        )
    }
}

pub struct ZkVerify{
    n : u128,
    cipher : u128,
    u1 : pallas::Affine,
    y : pallas::Affine,
    z : u128,
    u2 : u128,
    u3 : u128,
    e : u128,
    s1 : u128,
    s2 : u128,
    s3 : u128,
    h1 : u128,
    h2 : u128,
}

impl ZkVerify{
    pub fn zk_verify(&self) -> bool{
        let n = self.n;
        let w = self.cipher;
        let e = self.e;
        let y= self.y;
        let z= self.z;
        let s1 = self.s1;
        let s2 = self.s2;
        let s3 = self.s3;
        let u1 = self.u1;
        let u2 = self.u2;
        let u3 = self.u3;
        let h1 = self.h1;
        let h2 = self.h2;

        let generator = pallas::Affine::generator();
        let tau = n+1;

        let e_neg = pallas::Scalar::from_u128(e).neg();
        let s1g = pallas::Affine::mul(generator, pallas::Scalar::from_u128(s1)).to_affine();
        let ye_neg = pallas::Affine::mul(y, e_neg).to_affine();
        let s1g_plus_ye_neg = pallas::Affine::add(s1g, ye_neg).to_affine();
        let result1 = pallas::Affine::eq(&s1g_plus_ye_neg, &u1);

        let v1 = pow_and_mod(tau, s1, n*n);
        let v2 = pow_and_mod(s2, n, n*n);
        let w_inv = find_mod_inverse(w,n*n);
        let check = (w * w_inv) % (n*n);
        if check != 1{
            println!("no inv");
        }
        let v3 = pow_and_mod(w_inv, e, n*n);
        let v4 = (v1 * v2) % (n * n);
        let v5 = (v3 * v4) % (n * n);
        let result2 = u128::eq(&u2, &v5);
 
        let z2 = z % n;
        let v6 = pow_and_mod(h1, s1, n);
        let v7 = pow_and_mod(h2, s3, n);
        let neg_e3 = n - (e+1);
        let v8 = pow_and_mod(z2, neg_e3, n);
        let v9 = (v6 * v7) % n;
        let v = (v8 * v9) % n;
        let result3 = u128::eq(&u3, &v);

        let result = result1 && result2 && result3;
    
        result
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn zk_verify_test(){
        let n = 3517;
        let tau = n + 1;
        let secret = generate_random_u128_in_range(1, std::u8::MAX as u128);

        let generator = pallas::Affine::generator();
        let y = pallas::Affine::mul(generator, pallas::Scalar::from_u128(secret)).to_affine();
        let range_q = generate_random_u128_in_range(1, std::u16::MAX as u128);
        let r = generate_random_u128_in_range(1, range_q*range_q*range_q*n);

        let w1 = pow_and_mod(tau, secret, n*n);
        let w2 = pow_and_mod(r, n, n*n);
        let w = (w1 * w2) % (n*n);

        let prover_side = PrepareZk{
            n : n,
            secret : secret,
            r : r,
        };

        let ([u1,y],[z,u2,u3,e,s1,s2,s3,h1,h2]) = prover_side.prepare_zk_verify();

        let verifier_side = ZkVerify{
            n : n,
            cipher : w,
            u1 : u1,
            y : y,
            z : z,
            u2 : u2,
            u3 : u3,
            e : e,
            s1 : s1,
            s2 : s2,
            s3 : s3,
            h1 : h1,
            h2 : h2,
        };

        let result = verifier_side.zk_verify();

        assert_eq!(result,true);
    }
}

// fn main(){
//     let n = 3517;
//     let tau = n + 1;
//     let secret = generate_random_u128_in_range(1, std::u8::MAX as u128);

//     let generator = pallas::Affine::generator();
//     let y = pallas::Affine::mul(generator, pallas::Scalar::from_u128(secret)).to_affine();
//     let range_q = generate_random_u128_in_range(1, std::u16::MAX as u128);

//     let h1 = generate_random_u128_in_range(1, std::u64::MAX as u128);
//     let h2 = generate_random_u128_in_range(1, std::u64::MAX as u128);
//     let a = generate_random_u128_in_range(1, range_q);
//     let b = generate_random_u128_in_range(1, n);
//     let p = generate_random_u128_in_range(1, range_q*n);
//     let r = generate_random_u128_in_range(1, range_q*range_q*range_q*n);

//     let w1 = pow_and_mod(tau, secret, n*n);
//     let w2 = pow_and_mod(r, n, n*n);
//     let w = (w1 * w2) % (n*n);//cipher

//     let z1 = pow_and_mod(h1, secret, n);
//     let z2 = pow_and_mod(h2, p, n);
//     let z = (z1 * z2) % n;
//     let u1 = pallas::Affine::mul(generator, pallas::Scalar::from_u128(a)).to_affine();
//     let u21  = pow_and_mod(tau, a, n*n);
//     let u22  = pow_and_mod(b, n, n*n);
//     let u2 = (u21 * u22) % (n*n);
//     let u31 = pow_and_mod(h1, a, n);
//     let u32 = pow_and_mod(h2, r, n);
//     let u3 = (u31 * u32) % n;
//     let e = generate_random_u128_in_range(1, std::u8::MAX as u128);
//     let s1 = (e * secret) + a;
//     let s20 = pow_and_mod(r, e, n*n);
//     let s2 = s20 * b;
//     let s3 = (e * p) + r;

//     let e_neg = pallas::Scalar::from_u128(e).neg();
//     let s1g = pallas::Affine::mul(generator, pallas::Scalar::from_u128(s1)).to_affine();
//     let ye_neg = pallas::Affine::mul(y, e_neg).to_affine();
//     let s1g_plus_ye_neg = pallas::Affine::add(s1g, ye_neg).to_affine();
//     let result1 = pallas::Affine::eq(&s1g_plus_ye_neg, &u1);
//     println!("{}",result1);

//     let v1 = pow_and_mod(tau, s1, n*n);
//     let v2 = pow_and_mod(s2, n, n*n);
//     let w_inv = find_mod_inverse(w,n*n);
//     let check = (w * w_inv) % (n*n);
//     if check != 1{
//         println!("no inv");
//     }
//     let v3 = pow_and_mod(w_inv, e, n*n);
//     let v4 = (v1 * v2) % (n * n);
//     let v5 = (v3 * v4) % (n * n);
//     let result2 = u128::eq(&u2, &v5);
//     println!("{}",result2);

//     let z2 = z % n;
//     let v6 = pow_and_mod(h1, s1, n);
//     let v7 = pow_and_mod(h2, s3, n);
//     let neg_e3 = n - (e+1);
//     let v8 = pow_and_mod(z2, neg_e3, n);
//     let v9 = (v6 * v7) % n;
//     let v = (v8 * v9) % n;
//     let result3 = u128::eq(&u3, &v);
//     println!("{}",result3);
// }