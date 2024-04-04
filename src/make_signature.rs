use crate::generate_random_u128_in_range;
use myp::{Encrypt,EncryptAddMut,Decrypt};
use pasta_curves::{group::ff::PrimeField, pallas};
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
    others_mta_pub_n : Vec<u128>,//lens = n
    others_cipher_k : Vec<u128>,//lens = n
    selfw : Vec<u128>,//lens = n
}
//lens =(n*n, n*n)
impl SecondStep {
    fn cipher_k(&self) -> (Vec<u128>, Vec<u128>){
        // let random_unm = generate_random_u128_in_range(1,  self.mta_pub_n);
        let mut add_num_neg = Vec::new();
        let mut random_num2 = Vec::new();
        for _ in self.others_mta_pub_n.iter(){
            for (i, j) in self.others_mta_pub_n.iter().zip(&self.selfw){
                add_num_neg.push(generate_random_u128_in_range(1, *j));
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
    pri_p : Vec<u128>,//n
    pri_q : Vec<u128>,//n
    cipher : Vec<u128>,//n*n
}
//out put = n*(n-1)
impl ThirdStep {
    fn decrypt_cipher(&self) -> Vec<u128>{
        let mut pri_keyp = Vec::new();
        let mut pri_keyq = Vec::new();
        for (p, q) in self.pri_p.iter().zip(&self.pri_q){
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
    plain_text : Vec<pallas::Scalar>,//n*n
    selfk : Vec<pallas::Scalar>,//n
    selfw : Vec<pallas::Scalar>,
    add_num_neg : Vec<pallas::Scalar>,//n*n
    message : pallas::Scalar,
    r : pallas::Scalar,
}
//out put n
impl FourthStep {
    fn combine(&self) -> Vec<pallas::Scalar>{
        let mut sharding_signature = Vec::new();
        //k1w1
        for (i ,j) in self.selfk.iter().zip(&self.selfw){
            sharding_signature.push(pallas::Scalar::mul(i, j));
        }
        
        //k1w1 + plentext - add_num_neg = (k1 + k2 + ....) * (w1 +w2 + ....)
        //mul r
        let mut player = 0;
        let mut raw = 0;
        for i in &self.plain_text{
            if player != raw{
                sharding_signature[player] = pallas::Scalar::add(&sharding_signature[player], i);
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
                sharding_signature[raw] = pallas::Scalar::sub(&sharding_signature[raw], j);
            }
            player+=1;
            if player == self.selfk.len(){
                player=0;
                raw+=1;
            }
        }
        
        //mk + rkx
        player = 0;
        let mut mk : pallas::Scalar;
        for i in &self.selfk{
            sharding_signature[player] = pallas::Scalar::mul(&sharding_signature[player], &self.r);
            mk = pallas::Scalar::mul(i, &self.message);
            sharding_signature[player] = pallas::Scalar::add(&sharding_signature[player], &mk);
            player+=1;
        }
        sharding_signature
    }
}

pub struct MakeSignature{
    pub selfk : Vec<u128>,
    pub mta_pub_n : Vec<u128>,
    pub selfw : Vec<u128>,//lens = n
    pub pri_p : Vec<u128>,//n
    pub pri_q : Vec<u128>,//n
    pub message : u128,
    pub r : pallas::Scalar,
}

impl MakeSignature{
    pub fn make_signature(&self) -> Vec<pallas::Scalar> {
        let step1 = FirstStep{
            selfk : self.selfk.clone(),
            mta_pub_n : self.mta_pub_n.clone(),
        };
        let cipher_k = step1.encrypt_k();

        let step2 = SecondStep{
            others_mta_pub_n : self.mta_pub_n.clone(),
            others_cipher_k : cipher_k,
            selfw : self.selfw.clone(),
        };
        let (cipher_k2w1_plus_rand, add_nun_key) =  step2.cipher_k();

        let step3 = ThirdStep{
            pri_p : self.pri_p.clone(),
            pri_q : self.pri_q.clone(),
            cipher : cipher_k2w1_plus_rand,
            };
        let plentext = step3.decrypt_cipher();

        let mut plentext2 = Vec::new();
        for i in plentext{
            plentext2.push(pallas::Scalar::from_u128(i));
        }
        let mut kfq = Vec::new();
        for i in &self.selfk{
            kfq.push(pallas::Scalar::from_u128(*i));
        }
        let mut wfq = Vec::new();
        for i in &self.selfw{
            wfq.push(pallas::Scalar::from_u128(*i));
        }
        let mut add_nun_key2 = Vec::new();
        for i in add_nun_key{
            add_nun_key2.push(pallas::Scalar::from_u128(i));
        }

        let step4 = FourthStep{
            plain_text : plentext2,
            selfk : kfq,
            selfw : wfq,
            add_num_neg : add_nun_key2,
            message : pallas::Scalar::from_u128(self.message),
            r : self.r,
        };
        let result = step4.combine();

        result
    }
}

pub struct MakeSignature2{
    pub message :  pallas::Scalar,
    pub k : u128,
    pub r : pallas::Scalar,
    pub w : pallas::Scalar,
}

impl MakeSignature2{
    pub fn make_signature2(&self) -> pallas::Scalar{
        let k = pallas::Scalar::from_u128(self.k);
        let wr = pallas::Scalar::mul(&self.w, &self.r);
        let m_plus_wr = pallas::Scalar::add(&self.message,&wr);
        let k_m_plus_wr = pallas::Scalar::mul(&m_plus_wr,& k);

        k_m_plus_wr
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn text(){
        let ap = 37057;
        let bp = 45497;
        let cp = 58237;

        let aq = 55021;
        let bq = 61363;
        let cq = 50129;

        let an = ap*aq;
        let bn = bp*bq;
        let cn = cp*cq;

        let ak = generate_random_u128_in_range(1, std::u8::MAX as u128);
        let bk = generate_random_u128_in_range(1, std::u8::MAX as u128);
        let ck = generate_random_u128_in_range(1, std::u8::MAX as u128);

        let aw = generate_random_u128_in_range(1, std::u8::MAX as u128);
        let bw = generate_random_u128_in_range(1, std::u8::MAX as u128);
        let cw = generate_random_u128_in_range(1, std::u8::MAX as u128);

        let message = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let r = generate_random_u128_in_range(1, std::u64::MAX as u128);
        let r2 = pallas::Scalar::from_u128(r);

        let instence = MakeSignature{
            selfk : [ak, bk, ck].to_vec(),
            mta_pub_n : [an, bn, cn].to_vec(),
            selfw : [aw, bw, cw].to_vec(),
            pri_p : [ap, bp, cp].to_vec(),
            pri_q : [aq, bq, cq].to_vec(),
            message : message,
            r : r2,
        };

        let result = instence.make_signature();
        let mut ans = pallas::Scalar::zero();
        for i in &result{
            ans = pallas::Scalar::add(&ans, i);
        } 
        let check = message * (ak + bk + ck) + r * (ak + bk + ck) * (aw + bw + cw);
        let check2 = pallas::Scalar::from_u128(check);
        assert_eq!(check2, ans);
    }
}