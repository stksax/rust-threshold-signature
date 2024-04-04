use rand::Rng;

pub fn generate_random_u128_in_range(min: u128, max: u128) -> u128 {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(min..=max);
    random_value
}

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

pub fn invmod(a: u128, p: u128) -> u128 {
    if a == 0 {
        panic!("0 has no inverse mod {}", p);
    }
    let mut r = a;
    let mut d = 1;

    for _ in 0..p {
        d = ((p / r + 1) * d) % p;
        r = (d * a) % p;

        if r == 1 {
            break;
        }
    }

    if r != 1 {
        panic!("{} has no inverse mod {}", a, p);
    }

    d
}

pub struct Encrypt {
    pub mta_pub_n : u128,
    pub rand : u128,
    pub message : u128,
}

impl Encrypt{
    pub fn encrypt(self) -> u128{
        let x = pow_and_mod(self.rand, self.mta_pub_n, self.mta_pub_n*self.mta_pub_n);
        let cipher = (pow_and_mod(self.mta_pub_n + 1, self.message, self.mta_pub_n*self.mta_pub_n) * x) % (self.mta_pub_n*self.mta_pub_n);
        cipher
    }
}

pub struct EncryptAddMut {
    pub mta_pub_n : u128,
    pub cipher : u128,
    pub add_num : u128,
    pub mut_num : u128,
    pub rand : u128,
}

impl EncryptAddMut {
    pub fn mut_and_add(&self) -> u128{
        let mut cipher = pow_and_mod(self.cipher, self.mut_num, self.mta_pub_n*self.mta_pub_n);

        let value1 = pow_and_mod(self.mta_pub_n + 1, self.add_num, self.mta_pub_n*self.mta_pub_n);
        let value2 = pow_and_mod(self.rand , self.mta_pub_n, self.mta_pub_n*self.mta_pub_n);
        let add_value = (value1 * value2) % (self.mta_pub_n*self.mta_pub_n);
        cipher = (add_value * cipher) % (self.mta_pub_n*self.mta_pub_n);

        cipher
    }
}

pub struct Decrypt {
    pub pri_p : u128,
    pub pri_q : u128,
    pub cipher : u128,
}

impl Decrypt{
    pub fn decrypt(self) -> u128{
        let mta_pubkey_n = self.pri_p * self.pri_q;
        let g = (self.pri_p * self.pri_q) + 1;
        let lcm = num_integer::lcm(self.pri_p - 1, self.pri_q - 1);
        let gs = pow_and_mod(g, lcm, mta_pubkey_n * mta_pubkey_n);
        let l = (gs-1)/mta_pubkey_n;
        let u = invmod(l, mta_pubkey_n);

        let cs = pow_and_mod(self.cipher, lcm, mta_pubkey_n * mta_pubkey_n);
        let cl = (cs-1)/mta_pubkey_n;
        let plein_text = (cl * u) % mta_pubkey_n;
        plein_text
    }
}



