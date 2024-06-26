use std::fmt::Debug;
use std::hash::Hasher;
use std::ops::{Add, Mul};
use ff::Field;
use pasta_curves::group::cofactor::CofactorCurveAffine;
use pasta_curves::group::ff::PrimeField;
use pasta_curves::group::{Curve, Group};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value, Chip},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2_gadgets::utilities::{lookup_range_check::LookupRangeCheckConfig, UtilitiesInstructions};
use halo2_gadgets::ecc::*;
use halo2_gadgets::sinsemilla::chip::{SinsemillaChip, SinsemillaConfig};
use lazy_static::lazy_static;
use pasta_curves::pallas;
use halo2_gadgets::sinsemilla::{HashDomains, CommitDomains};
use halo2_gadgets::sinsemilla;
use std::collections::hash_map::DefaultHasher;
use halo2_gadgets::ecc::{
    chip::{
        find_zs_and_us, BaseFieldElem, EccChip, EccConfig, FixedPoint, FullScalar, ShortScalar,
        H, NUM_WINDOWS, NUM_WINDOWS_SHORT,
    },
    FixedPoints,
};

#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) struct TestFixedBases;
#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) struct FullWidth(pallas::Affine, &'static [(u64, [pallas::Base; H])]);
#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) struct BaseField;
#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) struct Short;

pub(crate) const PERSONALIZATION: &str = "MerkleCRH";

lazy_static! {
    static ref BASE: pallas::Affine = pallas::Point::generator().to_affine();
    static ref ZS_AND_US: Vec<(u64, [pallas::Base; H])> =
        find_zs_and_us(*BASE, NUM_WINDOWS).unwrap();
    static ref ZS_AND_US_SHORT: Vec<(u64, [pallas::Base; H])> =
        find_zs_and_us(*BASE, NUM_WINDOWS_SHORT).unwrap();
    static ref COMMIT_DOMAIN: sinsemilla::primitives::CommitDomain =
        sinsemilla::primitives::CommitDomain::new(PERSONALIZATION);
    static ref Q: pallas::Affine = *BASE;
    static ref R: pallas::Affine = *BASE;
    static ref R_ZS_AND_US: Vec<(u64, [pallas::Base; H])> =
        find_zs_and_us(*R, NUM_WINDOWS).unwrap();
}

impl FullWidth {
    pub(crate) fn from_pallas_generator() -> Self {
        FullWidth(*BASE, &ZS_AND_US)
    }

    pub(crate) fn from_parts(
        base: pallas::Affine,
        zs_and_us: &'static [(u64, [pallas::Base; H])],
    ) -> Self {
        FullWidth(base, zs_and_us)
    }
}

impl FixedPoint<pallas::Affine> for FullWidth {
    type FixedScalarKind = FullScalar;

    fn generator(&self) -> pallas::Affine {
        self.0
    }

    fn u(&self) -> Vec<[[u8; 32]; H]> {
        self.1
            .iter()
            .map(|(_, us)| {
                [
                    us[0].to_repr(),
                    us[1].to_repr(),
                    us[2].to_repr(),
                    us[3].to_repr(),
                    us[4].to_repr(),
                    us[5].to_repr(),
                    us[6].to_repr(),
                    us[7].to_repr(),
                ]
            })
            .collect()
    }

    fn z(&self) -> Vec<u64> {
        self.1.iter().map(|(z, _)| *z).collect()
    }
}

impl FixedPoint<pallas::Affine> for BaseField {
    type FixedScalarKind = BaseFieldElem;

    fn generator(&self) -> pallas::Affine {
        *BASE
    }

    fn u(&self) -> Vec<[[u8; 32]; H]> {
        ZS_AND_US
            .iter()
            .map(|(_, us)| {
                [
                    us[0].to_repr(),
                    us[1].to_repr(),
                    us[2].to_repr(),
                    us[3].to_repr(),
                    us[4].to_repr(),
                    us[5].to_repr(),
                    us[6].to_repr(),
                    us[7].to_repr(),
                ]
            })
            .collect()
    }

    fn z(&self) -> Vec<u64> {
        ZS_AND_US.iter().map(|(z, _)| *z).collect()
    }
}

impl FixedPoint<pallas::Affine> for Short {
    type FixedScalarKind = ShortScalar;

    fn generator(&self) -> pallas::Affine {
        *BASE
    }

    fn u(&self) -> Vec<[[u8; 32]; H]> {
        ZS_AND_US_SHORT
            .iter()
            .map(|(_, us)| {
                [
                    us[0].to_repr(),
                    us[1].to_repr(),
                    us[2].to_repr(),
                    us[3].to_repr(),
                    us[4].to_repr(),
                    us[5].to_repr(),
                    us[6].to_repr(),
                    us[7].to_repr(),
                ]
            })
            .collect()
    }

    fn z(&self) -> Vec<u64> {
        ZS_AND_US_SHORT.iter().map(|(z, _)| *z).collect()
    }
}

impl FixedPoints<pallas::Affine> for TestFixedBases {
    type FullScalar = FullWidth;
    type ShortScalar = Short;
    type Base = BaseField;
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct TestHashDomain;
impl HashDomains<pallas::Affine> for TestHashDomain {
    fn Q(&self) -> pallas::Affine {
        *Q
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct TestCommitDomain;
impl CommitDomains<pallas::Affine, TestFixedBases, TestHashDomain> for TestCommitDomain {
    fn r(&self) -> FullWidth {
        FullWidth::from_parts(*R, &R_ZS_AND_US)
    }

    fn hash_domain(&self) -> TestHashDomain {
        TestHashDomain
    }
}

#[derive(Default)]
struct Eddsa {
    commitment : pallas::Affine,
    pub_key : pallas::Affine,
    e : pallas::Scalar,
    s : pallas::Scalar,
}

#[allow(non_snake_case)]
impl Circuit<pallas::Base> for Eddsa {
    type Config = (
        EccConfig<TestFixedBases>,
        SinsemillaConfig<TestHashDomain, TestCommitDomain, TestFixedBases>,
    );
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let lookup_table = meta.lookup_table_column();
        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        let lookup = (
            lookup_table,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
        );

        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], lookup_table);
        let ecc_config = EccChip::<TestFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);
        let configs = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[2],
            lagrange_coeffs[0],
            lookup,
            range_check,
        );
        (ecc_config, configs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = EccChip::construct(config.0.clone());
        let column  = chip.config().advices[0];     
        SinsemillaChip::<TestHashDomain, TestCommitDomain, TestFixedBases>::load(
            config.1.clone(),
            &mut layouter,
        )?;

        let pub_key = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "public key"), 
            Value::known(self.pub_key),
        )?;

        let fp = pallas::Base::from_repr(self.e.to_repr()).unwrap();
        let base = chip.load_private(
            layouter.namespace(|| "e"), 
            column, 
            Value::known(fp),
        )?;

        let scalar = ScalarVar::from_base(
            chip.clone(), 
            layouter.namespace(|| "e as scalar"), 
            &base,
        )?;

        let (epub,_) = NonIdentityPoint::mul(
            &pub_key, 
            layouter.namespace(|| "e * pub key"), 
            scalar,
        )?;

        let p3 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "k * G"), 
            Value::known(self.commitment),
        )?;

        let p5 = Point::add(
            &p3, 
            layouter.namespace(|| "(k * G) + (e * pubkey)"), 
            &epub,
        )?;

        let affine_generator = pallas::Affine::generator();
        let s = pallas::Affine::mul(affine_generator, self.s).to_affine();
        let p4 =  Point::new(
            chip.clone(), 
            layouter.namespace(|| "(k + e * pri key) * G"), 
            Value::known(s),
        )?;
        
        let result = Point::constrain_equal(
            &p4, 
            layouter.namespace(|| "(k + e * pri key) * G == (k * G) + (e * pubkey)"), 
            &p5,
        );
        result
    }
}

#[derive(Default)]
struct VerifyKey {
    pub_key_send_before : pallas::Affine,
    pub_key_submit : [pallas::Affine;3],
}

#[allow(non_snake_case)]
impl Circuit<pallas::Base> for VerifyKey {
    type Config = (
        EccConfig<TestFixedBases>,
        SinsemillaConfig<TestHashDomain, TestCommitDomain, TestFixedBases>,
    );
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let lookup_table = meta.lookup_table_column();
        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        let lookup = (
            lookup_table,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
        );

        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], lookup_table);
        let ecc_config = EccChip::<TestFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);
        let configs = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[2],
            lagrange_coeffs[0],
            lookup,
            range_check,
        );
        (ecc_config, configs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = EccChip::construct(config.0.clone());
        let column  = chip.config().advices[0];     
        SinsemillaChip::<TestHashDomain, TestCommitDomain, TestFixedBases>::load(
            config.1.clone(),
            &mut layouter,
        )?;

        let pub_key_send_before = Point::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key_send_before"), 
            Value::known(self.pub_key_send_before),
        )?;

        let pub_key_share1 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key_share1"), 
            Value::known(self.pub_key_submit[0]),
        )?;

        let pub_key_share2 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key_share2"), 
            Value::known(self.pub_key_submit[1]),
        )?;

        let pub_key_share3 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key_share3"), 
            Value::known(self.pub_key_submit[2]),
        )?;
        
        let add_point = Point::add(
            &pub_key_share1, 
            layouter.namespace(|| "pub_key_share1 + pub_key_share2"), 
            &pub_key_share2, 
        )?;

        let add_point = Point::add(
            &add_point, 
            layouter.namespace(|| "pub_key_share1 + pub_key_share2 + pub_key_share3"), 
            &pub_key_share3, 
        )?;

        let result = Point::constrain_equal(
            &pub_key_send_before, 
            layouter.namespace(|| "pub_key_share1 + pub_key_share2 + pub_key_share3 == pub_key"), 
            &add_point,
        );
        result
    }
}

#[derive(Default)]
struct MyCircuit {
    commitment : [pallas::Affine;3],
    pub_key : [pallas::Affine;3],
    e : pallas::Scalar,
    s : [pallas::Scalar;3],
}

#[allow(non_snake_case)]
impl Circuit<pallas::Base> for MyCircuit {
    type Config = (
        EccConfig<TestFixedBases>,
        SinsemillaConfig<TestHashDomain, TestCommitDomain, TestFixedBases>,
    );
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let lookup_table = meta.lookup_table_column();
        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
      
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        let lookup = (
            lookup_table,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
        );

        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], lookup_table);
        let ecc_config = EccChip::<TestFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);
        let configs = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[2],
            lagrange_coeffs[0],
            lookup,
            range_check,
        );
        (ecc_config, configs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = EccChip::construct(config.0.clone());
        let column  = chip.config().advices[0];     
        SinsemillaChip::<TestHashDomain, TestCommitDomain, TestFixedBases>::load(
            config.1.clone(),
            &mut layouter,
        )?;

        let input_pub_key0 = Value::known(self.pub_key[0]);
        let input_pub_key1 = Value::known(self.pub_key[1]);
        let input_pub_key2 = Value::known(self.pub_key[2]);

        let pub_key0 = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "singal pub key0"), 
            input_pub_key0,
        )?;

        let pub_key1 = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "singal pub key1"), 
            input_pub_key1,
        )?;

        let pub_key2 = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "singal pub key2"), 
            input_pub_key2,
        )?;

        let pub_key = NonIdentityPoint::add_incomplete(
            &pub_key0, 
            layouter.namespace(|| "pub key0 + pub key1"), 
            &pub_key1, 
        )?;

        let pub_key = NonIdentityPoint::add_incomplete(
            &pub_key, 
            layouter.namespace(|| "pub key0 + pub key1 + pub_key2"), 
            &pub_key2, 
        )?;

        let fp = pallas::Base::from_repr(self.e.to_repr()).unwrap();
        let base = chip.load_private(
            layouter.namespace(|| "e"), 
            column, 
            Value::known(fp),
        )?;

        let scalar = ScalarVar::from_base(
            chip.clone(), 
            layouter.namespace(|| "e as scalar"), 
            &base,
        )?;

        let (epub,_) = NonIdentityPoint::mul(
            &pub_key, 
            layouter.namespace(|| "e * pub key"), 
            scalar,
        )?;

        let input_commitment0 = Value::known(self.commitment[0]);
        let input_commitment1 = Value::known(self.commitment[1]);
        let input_commitment2 = Value::known(self.commitment[2]);

        let singal_commitment0 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "singal_commitment0"), 
            input_commitment0,
        )?;

        let singal_commitment1 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "singal_commitment1"), 
            input_commitment1,
        )?;

        let singal_commitment2 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "singal_commitment2"), 
            input_commitment2,
        )?;

        let commitment = Point::add(
            &singal_commitment0, 
            layouter.namespace(|| "commitment0 + commitment1"), 
            &singal_commitment1,
        )?;

        let commitment = Point::add(
            &commitment, 
            layouter.namespace(|| "commitment0 + commitment1 + commitment2"), 
            &singal_commitment2,
        )?;

        let epub_plus_commitment = Point::add(
            &commitment, 
            layouter.namespace(|| "commitment + (e * pub key)"), 
            &epub,
        )?;


        let affine_generator = pallas::Affine::generator();
        let input_s_sum = self.s[0].add(self.s[1]);
        let input_s_sum = input_s_sum.add(self.s[2]);
        let s: pasta_curves::EpAffine = pallas::Affine::mul(affine_generator, input_s_sum).to_affine();


        let s_mul_G =  Point::new(
            chip.clone(), 
            layouter.namespace(|| "s * G"), 
            Value::known(s),
        )?;
       
        let result = Point::constrain_equal(
            &s_mul_G, 
            layouter.namespace(|| "s * G == commitment + e * pubkey"), 
            &epub_plus_commitment,
        );

        result
    }
    
}

pub fn pre_compute(
    pri : pallas::Scalar,
    input_r : pallas::Scalar,
    message : u128,
) -> (pallas::Affine, pallas::Scalar){
    let affine_generator = pallas::Affine::generator();
    let r = pallas::Affine::mul(affine_generator, input_r).to_affine();
    let mut hasher = DefaultHasher::new();
    hasher.write_u128(message);
    let hash_value = hasher.finish() as u128;
    let temp = pallas::Scalar::mul(&pri, &pallas::Scalar::from_u128(hash_value));
    let s = pallas::Scalar::add(&input_r, &temp);
    (r, s)
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{generate_random_u128_in_range,Input,CalculatePubKey,CalculatePriKey,CollectOutputKeyShare};
    #[test]
fn eddsa_test() {
    //there are 5 player join teh key generation
    let player1 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let player2 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let player3 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let player4 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let player5 = generate_random_u128_in_range(1, std::u64::MAX as u128);

    let input1 = Input{
        key_share : player1,
        rand_num : 379278,
        output_max : 5,
        output_min : 3,
    };
    let result1 = input1.output_key_share();

    let input2 = Input{
        key_share : player2,
        rand_num : 4812738974,
        output_max : 5,
        output_min : 3,
    };
    let result2 = input2.output_key_share();

    let input3 = Input{
        key_share : player3,
        rand_num : 43217,
        output_max : 5,
        output_min : 3,
    };
    let result3 = input3.output_key_share();

    let input4 = Input{
        key_share : player4,
        rand_num : 12343432,
        output_max : 5,
        output_min : 3,
    };
    let result4 = input4.output_key_share();

    let input5 = Input{
        key_share : player5,
        rand_num : 1234546,
        output_max : 5,
        output_min : 3,
    };
    let result5 = input5.output_key_share();

    let mut user_vec = Vec::new();
    user_vec.extend(result1);
    user_vec.extend(result2);
    user_vec.extend(result3);
    user_vec.extend(result4);
    user_vec.extend(result5);

    let user1 = CollectOutputKeyShare{
        key_share : user_vec.clone(),
        member : 5,
        self_num : 1,
    };
    let (user1_prikey_share_a, user1_pubkey_share) = user1.collect();
    let calculate_user1_prikey_share = CalculatePriKey {
        self_coefficient : 1,
        coefficient : [2,3],
        pri_key : user1_prikey_share_a,
    };
    let user1_prikey_share = calculate_user1_prikey_share.calculate();
    
    let user2 = CollectOutputKeyShare{
        key_share : user_vec.clone(),
        member : 5,
        self_num : 2,
    };
    let (user2_prikey_share_a, user2_pubkey_share) = user2.collect();
    let calculate_user2_prikey_share = CalculatePriKey {
        self_coefficient : 2,
        coefficient : [1,3],
        pri_key : user2_prikey_share_a,
    };
    let user2_prikey_share = calculate_user2_prikey_share.calculate();

    let user3 = CollectOutputKeyShare{
        key_share : user_vec.clone(),
        member : 5,
        self_num : 3,
    };
    let (user3_prikey_share_a, user3_pubkey_share) = user3.collect();
    let calculate_user3_prikey_share = CalculatePriKey {
        self_coefficient : 3,
        coefficient : [1,2],
        pri_key : user3_prikey_share_a,
    };
    let user3_prikey_share = calculate_user3_prikey_share.calculate();

    let pub_key_calaulate = CalculatePubKey {
        degree : 3,
        coefficient : [1,2,3].to_vec(),
        pub_key : [user1_pubkey_share, user2_pubkey_share, user3_pubkey_share].to_vec(),
    };
    //they make the public key
    let pub_key = pub_key_calaulate.calculate();
    //message is the thing they want to vote
    let message = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let generator = pallas::Affine::generator();
    
    let (r1, s1) = pre_compute(
        user1_prikey_share, 
        pallas::Scalar::random(rand::rngs::OsRng),
        message,
    );
    let user1_pubkey = generator.mul(user1_prikey_share).to_affine();

    let (r2, s2) = pre_compute(
        user2_prikey_share, 
        pallas::Scalar::random(rand::rngs::OsRng),
        message,
    );
    let user2_pubkey = generator.mul(user2_prikey_share).to_affine();

    let (r3, s3) = pre_compute(
        user3_prikey_share, 
        pallas::Scalar::random(rand::rngs::OsRng),
        message,
    );
    let user3_pubkey = generator.mul(user3_prikey_share).to_affine();
    
    let mut hasher = DefaultHasher::new();
    hasher.write_u128(message);
    let challange = hasher.finish() as u128;
    //this is just for make sure the user1_prikey_share add together is as our expect 
    //it doesn't exist in the real project 
    let check1 = player1 + player2 + player3 + player4 + player5;
    let check2 = user1_prikey_share + user2_prikey_share + user3_prikey_share;
    let check3 = pallas::Scalar::from_u128(check1);
    let pri_key_equal = pallas::Scalar::eq(&check2, &check3);
    assert_eq!(pri_key_equal,true);

    //here we use eddsa to verify singal signature from user1 to user3
    let k1 = 13;
    let circuit1 = Eddsa{
        s : s1,
        pub_key : user1_pubkey,
        commitment : r1,
        e : pallas::Scalar::from_u128(challange),
    };
    let prover1 = MockProver::run(k1, &circuit1, vec![]).unwrap();
    assert_eq!(prover1.verify(), Ok(()),"User1 had been hack");

    let k2 = 13;
    let circuit2 = Eddsa{
        s : s2,
        pub_key : user2_pubkey,
        commitment : r2,
        e : pallas::Scalar::from_u128(challange),
    };
    let prover2 = MockProver::run(k2, &circuit2, vec![]).unwrap();
    assert_eq!(prover2.verify(), Ok(()),"User2 had been hack");

    let k3 = 13;
    let circuit3 = Eddsa{
        s : s3,
        pub_key : user3_pubkey,
        commitment : r3,
        e : pallas::Scalar::from_u128(challange),
    };
    let prover3 = MockProver::run(k3, &circuit3, vec![]).unwrap();
    assert_eq!(prover3.verify(), Ok(()),"User3 had been hack");

    let k4 = 13;
    let circuit4 = VerifyKey{
        pub_key_send_before : pub_key,
        pub_key_submit : [user1_pubkey, user2_pubkey, user3_pubkey],
    };
    let prover4 = MockProver::run(k4, &circuit4, vec![]).unwrap();
    assert_eq!(prover4.verify(), Ok(()),"public key isn't right");
    
    let k = 13;
    let circuit = MyCircuit{
        s : [s1,s2,s3],
        pub_key : [user1_pubkey, user2_pubkey, user3_pubkey],
        commitment : [r1,r2,r3],
        e : pallas::Scalar::from_u128(challange),
    };
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()))
}
}