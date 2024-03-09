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

struct encrypt {
    input_n : i128,
    input_r : i128,
    message : i128,
}

impl encrypt{
    fn encrypt() -> i128{
        let x = pow_and_mod(self.input_r, self.input_n, self.input_n*self.input_n);
        let cipher = (pow_and_mod(self.input_n + 1, self.message, self.input_n*self.input_n) * x) % (self.input_n*self.input_n);
        cipher
    }
}

struct encrypt_add_mut {
    input_n : i128,
    cipher : i128,
    add_num : i128,
    mut_num : i128,
}

impl encrypt{
    fn encrypt_add() -> i128{
        let cipher = (self.add_num * self.cipher) % (self.input_n*self.input_n);
        cipher
    }
    fn encrypt_mut() -> i128{
        let cipher = modpow(self.cipher, self.mut_num, (self.input_n*self.input_n));
        cipher
    }
}

struct decrypt {
    input_p : i128,
    input_q : i128,
    input_n : i128,
    cipher : i128,
}

impl decrypt{
    fn decrypt() -> i128{
        let g = (self.input_p * self.input_q) + 1;
        let lcm = num_integer::lcm(self.input_p - 1, self.input_q - 1);
        let gs = pow_and_mod(g, lcm, (self.input_n*self.input_n));
        let l = (gs-1)/n;
        let u = invmod(l, self.input_n);

        let cs = pow_and_mod(self.cipher, lcm, (self.input_n*self.input_n));
        let cl = (cs-1)/n;
        let plein_text = (cl * u) % self.input_n;
        plein_text
    }
}

