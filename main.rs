pub fn pow_and_mod(base: i128, exponent: i128, modulus: i128) -> i128{
    let mut reslut: i128 = base;
    for i in 0..(exponent-1){
        reslut = reslut * base;
        reslut = reslut % modulus;
    }
    reslut
}

fn invmod(a: i128, p: i128) -> i128 {
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

struct Encrypt {
    input_n : i128,
    input_r : i128,
    message : i128,
}

impl Encrypt{
    fn encrypt(self) -> i128{
        let x = pow_and_mod(self.input_r, self.input_n, self.input_n*self.input_n);
        let cipher = (pow_and_mod(self.input_n + 1, self.message, self.input_n*self.input_n) * x) % (self.input_n*self.input_n);
        cipher
    }
}

struct Encrypt_add_mut {
    input_n : i128,
    cipher : i128,
    add_num : i128,
    mut_num : i128,
    rand : i128,
}

impl Encrypt_add_mut{
    fn encrypt_add(self) -> i128{
        let value1 = pow_and_mod(self.input_n + 1, self.add_num, self.input_n*self.input_n);
        let value2 = pow_and_mod(self.rand , self.input_n, self.input_n*self.input_n);
        let add_value = (value1 * value2) % (self.input_n*self.input_n);
        let cipher = (add_value * self.cipher) % (self.input_n*self.input_n);
        cipher
    }
    fn encrypt_mut(self) -> i128{
        let cipher = pow_and_mod(self.cipher, self.mut_num, self.input_n*self.input_n);
        cipher
    }
}

struct Decrypt {
    input_p : i128,
    input_q : i128,
    input_n : i128,
    cipher : i128,
}

impl Decrypt{
    fn decrypt(self) -> i128{
        let g = (self.input_p * self.input_q) + 1;
        let lcm = num_integer::lcm(self.input_p - 1, self.input_q - 1);
        let gs = pow_and_mod(g, lcm, self.input_n*self.input_n);
        let l = (gs-1)/self.input_n;
        let u = invmod(l, self.input_n);

        let cs = pow_and_mod(self.cipher, lcm, self.input_n*self.input_n);
        let cl = (cs-1)/self.input_n;
        let plein_text = (cl * u) % self.input_n;
        plein_text
    }
}

fn main(){
    let p: i128 = 1847;
    let q: i128 = 1721;
    let r: i128 = 1899;
    let m: i128 = 2000;
    let n = p*q;

    let encrypt = Encrypt{
        input_n : n,
        input_r : r,
        message : m,
    };
    let cipher = encrypt.encrypt();

    let reciver1 = Encrypt_add_mut{
        input_n : n,
        cipher : cipher,
        add_num : 1777,
        mut_num : 5,
        rand : 35267,
    };

    let new_cipher1 = reciver1.encrypt_mut();

    let reciver2 = Encrypt_add_mut{
        input_n : n,
        cipher : new_cipher1,
        add_num : 1777,
        mut_num : 5,
        rand : 35267,
    };
    let new_cipher2 = reciver2.encrypt_add();

    let decrypt = Decrypt{
        input_p : p,
        input_q : q,
        input_n : n,
        cipher : new_cipher2,
    };
    let plein_text = decrypt.decrypt();
    let ans = (plein_text - 1777)%n;
    println!("{}",ans);
    let range = std::i128::MAX;
    println!("{}",range);
   
}