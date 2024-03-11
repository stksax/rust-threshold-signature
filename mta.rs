pub fn pow_and_mod(base: u128, exponent: u128, modulus: u128) -> u128{
    let mut reslut: u128 = base;
    for i in 0..(exponent-1){
        reslut = reslut * base;
        reslut = reslut % modulus;
    }
    reslut
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
    pub input_r : u128,
    pub message : u128,
}

impl Encrypt{
    pub fn encrypt(self) -> u128{
        let x = pow_and_mod(self.input_r, self.mta_pub_n, self.mta_pub_n*self.mta_pub_n);
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
    pub fn encrypt_add(&self) -> u128{
        let value1 = pow_and_mod(self.mta_pub_n + 1, self.add_num, self.mta_pub_n*self.mta_pub_n);
        let value2 = pow_and_mod(self.rand , self.mta_pub_n, self.mta_pub_n*self.mta_pub_n);
        let add_value = (value1 * value2) % (self.mta_pub_n*self.mta_pub_n);
        let cipher = (add_value * self.cipher) % (self.mta_pub_n*self.mta_pub_n);
        cipher
    }
    pub fn encrypt_mut(&self) -> u128{
        let cipher = pow_and_mod(self.cipher, self.mut_num, self.mta_pub_n*self.mta_pub_n);
        cipher
    }
}

pub struct Decrypt {
    pub input_p : u128,
    pub input_q : u128,
    pub mta_pub_n : u128,
    pub cipher : u128,
}

impl Decrypt{
    pub fn decrypt(self) -> u128{
        let g = (self.input_p * self.input_q) + 1;
        let lcm = num_integer::lcm(self.input_p - 1, self.input_q - 1);
        let gs = pow_and_mod(g, lcm, self.mta_pub_n*self.mta_pub_n);
        let l = (gs-1)/self.mta_pub_n;
        let u = invmod(l, self.mta_pub_n);

        let cs = pow_and_mod(self.cipher, lcm, self.mta_pub_n*self.mta_pub_n);
        let cl = (cs-1)/self.mta_pub_n;
        let plein_text = (cl * u) % self.mta_pub_n;
        plein_text
    }
}


fn main(){
    let p: u128 = 1847;
    let q: u128 = 1721;
    let r: u128 = 1899;
    let m: u128 = 2000;
    let n = p*q;

    let encrypt = Encrypt{
        mta_pub_n : n,
        input_r : r,
        message : m,
    };
    let cipher = encrypt.encrypt();

    let reciver1 = EncryptAddMut{
        mta_pub_n : n,
        cipher : cipher,
        add_num : 1777,
        mut_num : 5,
        rand : 35267,
    };

    let new_cipher1 = reciver1.encrypt_mut();

    let reciver2 = EncryptAddMut{
        mta_pub_n : n,
        cipher : new_cipher1,
        add_num : 1777,
        mut_num : 5,
        rand : 35267,
    };
    let new_cipher2 = reciver2.encrypt_add();

    let decrypt = Decrypt{
        input_p : p,
        input_q : q,
        mta_pub_n : n,
        cipher : new_cipher2,
    };
    let plein_text = decrypt.decrypt();
    let ans = (plein_text - 1777)%n;
    println!("{}",ans);
    let range = std::u128::MAX;
    println!("{}",range);
   
}