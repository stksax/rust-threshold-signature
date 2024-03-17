mod mta; 
use mta::*;
use rand::Rng;
pub fn generate_random_u128_in_range(min: u128, max: u128) -> u128 {
    let mut rng = rand::thread_rng();
    let random_value = rng.gen_range(min..=max);
    random_value
}
//encrypt k and send
struct FirstStep{
    selfk : Vec<u128>,
    mta_pub_n : Vec<u128>,
}
//out put = n
impl FirstStep {
    fn encrypt_k(&self) -> Vec<u128>{
        let mut random_r = Vec::new();
        let mut iter = 0;
        for i in self.mta_pub_n.clone(){
            random_r.push(generate_random_u128_in_range(1, i));
        }

        let mut encrypt_instance = Vec::new();
        for (i, j) in self.selfk.iter().zip(&self.mta_pub_n) {
            encrypt_instance.push(Encrypt {
                mta_pub_n: *j,
                input_r: random_r[iter],
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
    others_mta_pub_n : Vec<u128>,//lens = n
    others_cipher_k : Vec<u128>,//lens = n
    selfw : Vec<u128>,//lens = n
    selfk : Vec<u128>,//lens = n
}
//lens =(n*n, n*n)
impl SecondStep {
    fn cipher_k(&self) -> (Vec<u128>, Vec<u128>){
        // let random_unm = generate_random_u128_in_range(1,  self.mta_pub_n);
        let mut add_num_neg = Vec::new();
        let mut random_num2 = Vec::new();
        for _ in self.others_mta_pub_n.iter(){
            for ((i, j), k) in self.others_mta_pub_n.iter().zip(&self.selfw).zip(&self.selfk){
                add_num_neg.push(generate_random_u128_in_range(1, *j * *k));
                random_num2.push(generate_random_u128_in_range(1, *i));
            }
        }
        
        let mut iter : usize = 0;
        let mut encrypt_instance = Vec::new();
        for k in &self.selfw {
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
    input_p : Vec<u128>,//n
    input_q : Vec<u128>,//n
    cipher : Vec<u128>,//n*n
}
//out put = n*(n-1)
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
                input_p : pri_keyp[iter],
                input_q : pri_keyq[iter],
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
    plain_text : Vec<u128>,//n*n
    selfk : Vec<u128>,//n
    selfw : Vec<u128>,
    add_num_neg : Vec<u128>,//n*n
    message : u128,
    r : u128,
}
//out put n
impl FourthStep {
    fn combine(&self) -> Vec<u128>{
        let mut sharding_signature = Vec::new();
        //k1w1
        for (i ,j) in self.selfk.iter().zip(&self.selfw){
            sharding_signature.push(*i * *j);
        }
        
        //k1w1 + plentext - add_num_neg = (k1 + k2 + ....) * (w1 +w2 + ....)
        //mul r
        let mut player = 0;
        let mut raw = 0;
        for i in &self.plain_text{
            if player != raw{
                sharding_signature[player] += *i;
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
            if player != raw{
                sharding_signature[raw] -= *j;
            }
            player+=1;
            if player == self.selfk.len(){
                player=0;
                raw+=1;
            }
        }
        
        //mk + rkx
        player = 0;
        for i in &self.selfk{
            sharding_signature[player] *= self.r;
            sharding_signature[player] += *i * self.message;
            player+=1;
        }
        sharding_signature
    }
}

fn main(){
    let ap = 2801;
    let aq = 2957;
    let bp = 2693;
    let bq = 2543;
    let cp = 2861;
    let cq = 3361;

    let an = 8282557;
    let bn = 6848299;
    let cn = 9615821;

    let ak = 423;
    let bk = 264;
    let ck = 784;

    let aw = 748;
    let bw = 374;
    let cw = 468;

    let step1 = FirstStep{
        selfk : [ak, bk, ck].to_vec(),
        mta_pub_n : [an, bn, cn].to_vec(),
    };
    
    let cipher_k = step1.encrypt_k();

    let step2 = SecondStep{
        others_mta_pub_n : [an, bn, cn].to_vec(),
        others_cipher_k : cipher_k,
        selfw : [aw, bw, cw].to_vec(),
        selfk : [ak, bk, ck].to_vec(),
    };

    //(n*n , n)
    let (cipher_k2w1_plus_rand, add_nun_key) =  step2.cipher_k();
 
    let step3 = ThirdStep{
    input_p : [ap, bp, cp].to_vec(),
    input_q : [aq, bq, cq].to_vec(),
    cipher : cipher_k2w1_plus_rand,
    };

    let plentext = step3.decrypt_cipher();

    let step4 = FourthStep{
        plain_text : plentext,
        selfk : [ak, bk, ck].to_vec(),
        selfw : [aw, bw, cw].to_vec(),
        add_num_neg : add_nun_key,
        message : 777,
        r : 333,
    };
    let result = step4.combine();
    let mut ans =0;
    for i in &result{
        ans+=i;
    } 
    println!("{:?}",result);
    println!("{:?}",ans);
    let check = 777 * (ak + bk + ck) + 333 * (ak + bk + ck) * (aw + bw + cw);
    println!("{:?}",check);
    println!("u128 max value: {}", std::u128::MAX);
}