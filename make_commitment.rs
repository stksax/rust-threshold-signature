use pasta_curves::pallas;

pub fn generate_random_u128_in_range(min: u128, max: u128) -> u128 {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(min..=max);
    random_value
}
//send cipher k
struct FirstStep{
    selfk : u128,
    input_n : u128,
}

impl FirstStep {
    fn cipher_k () -> u128{
        let encrypt_instance = Encrypt {
            input_n: self.input_n,
            input_r: generate_random_u128_in_range(0, self.input_n),
            message: self.selfk, 
        };

        let cipher_k = encrypt_instance.encrypt();
        cipher_k
    }
}
//recive cipherk and generate cipher(k2*r1+c1) and keep -c1 as key
struct SecondStep{
    input_n : u128,
    others_cipher_k : u128,
    selfr : u128,
}

impl SecondStep {
    fn cipher_k () -> [u128;2]{
        let encrypt_instance = EncryptAddMut {
            input_n : self.input_n,
            cipher : self.others_cipher_k,
            add_num : generate_random_u128_in_range(0,  self.input_n),
            mut_num : self.selfr,
            rand : generate_random_u128_in_range(0,  self.input_n),
        };

        let cipher_k2r1 = encrypt_instance.encrypt_mut();
        let cipher_k2r1_plus_c = encrypt_instance.encrypt_add();
        let add_num_neg = pallas::Scalar::from_u128(add_num).neg();
        [cipher_k2r1_plus_c,  add_num_neg]
    }
}
//decrypt cipher get k1*r2+c2
struct ThirdStep{
    input_p : u128,
    input_q : u128,
    input_n : u128,
    cipher : u128,
}

impl ThirdStep {
    fn decrypt_cipher () -> u128{
        let encrypt_instance = decrypt {
            input_p : self.input_p,
            input_q : self.input_q,
            input_n : self.input_n,
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
    add_num_neg : pallas::Scalar,
}

impl FourthStep {
    fn combine () -> (pallas::Scalar, pallas::Affine){
        let selfk = pallas::Scalar::from_u128(self.selfk);
        let selfr = pallas::Scalar::from_u128(self.selfr);
        let k1r1 = pallas::Scalar::mul(&selfk, &selfr);
        let k1r2 = pallas::Scalar::from_u128(self.plain_text);
        let k2r1 = self.add_num_neg;
        let add1 = pallas::Scalar::add(&k1r1, &k1r2);
        let sharding_commitment = pallas::Scalar::add(&add1, &k2r1).invert();

        let affine_generator = pallas::Affine::generator();
        let verify_point = pallas::Affine::mul(affine_generator, &selfr).to_affine();

        (sharding_commitment, verify_point)
    }
}

fn main() {
    let allice_selfk = 5646765;
    let allice_selfr = 3454667;
    let allice_input_p = 3163;
    let allice_input_q = 3541;
    let allice_mta_n = allice_input_p * allice_input_q;

    let bob_selfk = 688338;
    let bob_selfr = 46628;
    let bob_input_p = 3347;
    let bob_input_q = 2939;
    let bob_mta_n = bob_input_p * bob_input_q;
    

    let a_step_1 = FirstStep{
        selfk : allice_selfk,
        input_n : allice_mta_n,
    };
    let cipher_a1 = a_step_1.cipher_k();

    let b_step_1 = FirstStep{
        selfk : bob_selfk,
        input_n : bob_mta_n,
    };
    let cipher_b1 = b_step_1.cipher_k();

    let a_step_2 = SecondStep{
        input_n : allice_mta_n,
        others_cipher_k : cipher_b1,
        selfr : allice_selfr,
    };
    let [cipher_k2r1_plus_c1, neg_c1] = a_step_2.cipher_k();

    let b_step_2 = SecondStep{
        input_n : bob_mta_n,
        others_cipher_k : cipher_a1,
        selfr : bob_selfr,
    };
    let [cipher_k1r2_plus_c2, neg_c2] = b_step_2.cipher_k();

    let a_step_3 = ThirdStep{
        input_p : allice_input_p,
        input_q : allice_input_q,
        input_n : allice_mta_n,
        cipher : cipher_k2r1_plus_c1,
    };
    let plain_text_a = a_step_3.decrypt_cipher();

    let b_step_3 = ThirdStep{
        input_p : bob_input_p,
        input_q : bob_input_q,
        input_n : bob_mta_n,
        cipher : cipher_k1r2_plus_c2,
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
    let v2 = pallas::Scalar::from_u128(v1);
    let v3 = v2.invert();
    let v4 = pallas::Scalar::add(sharding_commitment_a, sharding_commitment_b);
    println!("{:?}",&v3);
    println!("{:?}",&v4);
}
//s = k (m + xr)