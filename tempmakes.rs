use std::fmt::Debug;
use std::ops::{Add, Mul};
use pasta_curves::arithmetic::CurveAffine;
mod tool;
use pasta_curves::group::cofactor::CofactorCurveAffine;
use tool::*;
mod make_commitment;
use make_commitment::make_commitment;
mod make_signature;
use make_signature::*;
mod key_generate;
use key_generate::*;
use halo2_proofs::arithmetic::Field;
use pasta_curves::group::ff:: PrimeField;
use pasta_curves::group::{Curve, Group, };
use num_integer::gcd;

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value, Chip},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2_gadgets::utilities::{lookup_range_check::LookupRangeCheckConfig, UtilitiesInstructions};
use halo2_gadgets::ecc::*;

use lazy_static::lazy_static;
use pasta_curves::pallas;

use halo2_gadgets::ecc::FixedPoint as EccFixedPoint;
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

lazy_static! {
    static ref BASE: pallas::Affine = pallas::Point::generator().to_affine();
    static ref ZS_AND_US: Vec<(u64, [pallas::Base; H])> =
        find_zs_and_us(*BASE, NUM_WINDOWS).unwrap();
    static ref ZS_AND_US_SHORT: Vec<(u64, [pallas::Base; H])> =
        find_zs_and_us(*BASE, NUM_WINDOWS_SHORT).unwrap();
}

impl FullWidth {
    pub(crate) fn from_pallas_generator() -> Self {
        FullWidth(*BASE, &ZS_AND_US)
    }

    // pub(crate) fn from_parts(
    //     base: pallas::Affine,
    //     zs_and_us: &'static [(u64, [pallas::Base; H])],
    // ) -> Self {
    //     FullWidth(base, zs_and_us)
    // }
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
#[derive(Default)]
struct MyCircuit {
    input_r : pallas::Scalar,
    input_s : pallas::Scalar,
    commitment : pallas::Affine,
    message : pallas::Scalar,
    pub_key :  pallas::Affine,
}

#[allow(non_snake_case)]
impl Circuit<pallas::Base> for MyCircuit {
    type Config = EccConfig<TestFixedBases>;
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
        // Shared fixed column for loading constants
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], lookup_table);
        EccChip::<TestFixedBases>::configure(meta, advices, lagrange_coeffs, range_check)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = EccChip::construct(config.clone());
        let column  = chip.config().advices[0];

        let fixedPoint = EccFixedPoint::from_inner(
            chip.clone(), 
            FullWidth::from_pallas_generator(),
        );

        let pub_key = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "P"), 
            Value::known(self.pub_key),
        )?;

        let commitment = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "P"), 
            Value::known(self.commitment),
        )?;

        let s_inv = pallas::Scalar::invert(&self.input_s).unwrap();
        let u1 = pallas::Scalar::mul(&self.message, &s_inv);

        let scalar1 = ScalarFixed::new(
            chip.clone(), 
            layouter.namespace(|| "P"), 
            Value::known(u1),
        )?;

        let (p1, _) = EccFixedPoint::mul(
            &fixedPoint, 
            layouter.namespace(|| "P"),  
            scalar1,
        )?;

        let u2 = pallas::Scalar::mul(&s_inv, &self.input_r);
        let u2repr = pallas::Scalar::to_repr(&u2);
        let u2_fp = pallas::Base::from_repr(u2repr).unwrap();
        let base = chip.load_private(
            layouter.namespace(|| "P"),  
            column, 
            Value::known(u2_fp),
        )?;

        let scalar2 =  ScalarVar::from_base(
            chip.clone(), 
            layouter.namespace(|| "P"), 
            &base,
        )?;

        let (p2, _) = NonIdentityPoint::mul(
            &pub_key, 
            layouter.namespace(|| "P"), 
            scalar2,
        )?;

        let p3 = Point::add(
            &p1, 
            layouter.namespace(|| "P"), 
            &p2,
        )?;

        let result = Point::constrain_equal(
            &p3, 
            layouter.namespace(|| "P"), 
            &commitment,
        );

        let ans1 = p3.extract_p();
        let ans2 = commitment.extract_p();
        println!("{:?}",&ans1);
        println!("{:?}",&ans2);
        result
    }
    
}

fn main() {
    let key_share1 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let key_share2 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let key_share3 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let key_share4 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let key_share5 = generate_random_u128_in_range(1, std::u64::MAX as u128);
    let input1 = Input{
        key_share : key_share1,
        rand_num : 379278,
        output_max : 5,
        output_min : 3,
    };
    let result1 = input1.output_key_share();
    
    let input2 = Input{
        key_share : key_share2,
        rand_num : 4812738974,
        output_max : 5,
        output_min : 3,
    };
    let result2 = input2.output_key_share();
    
    let input3 = Input{
        key_share : key_share3,
        rand_num : 43217,
        output_max : 5,
        output_min : 3,
    };
    let result3 = input3.output_key_share();
    
    let input4 = Input{
        key_share : key_share4,
        rand_num : 745,
        output_max : 5,
        output_min : 3,
    };
    let result4 = input4.output_key_share();
    
    let input5 = Input{
        key_share : key_share5,
        rand_num : 542,
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
    let (user1_prikey_share, user1_pubkey_share) = user1.collect();
        
    let user2 = CollectOutputKeyShare{
        key_share : user_vec.clone(),
        member : 5,
        self_num : 2,
    };
    let (user2_prikey_share, user2_pubkey_share) = user2.collect();
    
    let user3 = CollectOutputKeyShare{
        key_share : user_vec.clone(),
        member : 5,
        self_num : 3,
    };
    let (user3_prikey_share, user3_pubkey_share) = user3.collect();
    
    let pub_key_pre = CalculatePubKey {
        degree : 3,
        coefficient : [1,2,3].to_vec(),
        pub_key : [user1_pubkey_share, user2_pubkey_share, user3_pubkey_share].to_vec(),
    };

    let pub_key = pub_key_pre.calculate();

    let allice_selfk = 564;
    let allice_selfr = 345;
    let allice_mta_pri_p = 35023;//should in u64
    let allice_mta_pri_q = 46093;
    let allice_mta_pub_n = allice_mta_pri_p * allice_mta_pri_q;
    let mut gcd_check = gcd(allice_mta_pub_n, (allice_mta_pri_p-1)*(allice_mta_pri_q-1));
    if gcd_check!= 1{
        panic!("gcd !=1");
    }
    if (allice_selfk * allice_selfr) >= allice_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((allice_selfk*allice_selfr) < allice_mta_pub_n);

    let bob_selfk = 687;
    let bob_selfr = 466;
    let bob_mta_pri_p = 37889;//should in u64
    let bob_mta_pri_q = 55259;
    let bob_mta_pub_n = bob_mta_pri_p * bob_mta_pri_q;
    gcd_check = gcd(bob_mta_pub_n, (bob_mta_pri_p-1)*(bob_mta_pri_q-1));
    if gcd_check!= 1{
        panic!("gcd !=1");
    }
    if (bob_selfk * bob_selfr) >= bob_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((bob_selfk*bob_selfr) < bob_mta_pub_n);

    let chris_selfk = 745;
    let chris_selfr = 531;
    let chris_mta_pri_p = 64663;//should in u64
    let chris_mta_pri_q = 38113;
    let chris_mta_pub_n = chris_mta_pri_p * chris_mta_pri_q;
    gcd_check = gcd(chris_mta_pub_n, (chris_mta_pri_p-1)*(chris_mta_pri_q-1));
    if gcd_check!= 1{
        panic!("gcd !=1");
    }
    if (chris_selfk * chris_selfr) >= chris_mta_pub_n {
        panic!("k * r should < n , because it will mod n");
    }
    assert!((chris_selfk*chris_selfr) < chris_mta_pub_n);

    let selfk_vec = [allice_selfk, bob_selfk, chris_selfk].to_vec();
    let selfr_vec = [allice_selfr, bob_selfr, chris_selfr].to_vec();
    let mta_pub_n_vec = [allice_mta_pub_n, bob_mta_pub_n, chris_mta_pub_n].to_vec();
    let mta_pri_p_vec = [allice_mta_pri_p, bob_mta_pri_p, chris_mta_pri_p].to_vec();
    let mta_pri_q_vec = [allice_mta_pri_q, bob_mta_pri_q, chris_mta_pri_q].to_vec();

    let (sharding_commitment, verify_point)= make_commitment(
        selfk_vec.clone(),
        selfr_vec.clone(),
        mta_pub_n_vec.clone(),
        mta_pri_p_vec.clone(),
        mta_pri_q_vec.clone(),
    );
    
    let mut commitment = pallas::Scalar::zero();
    for i in &sharding_commitment{
        commitment = pallas::Scalar::add(&commitment, i);
    }

    let commitment2 = pallas::Scalar::invert(&commitment).unwrap();
    let mut commitment3 = pallas::Affine::identity();
    for i in &verify_point{
        commitment3 = pallas::Affine::add(commitment3, *i).to_affine();
    };
    let commitment4 = pallas::Affine::mul(commitment3, commitment2).to_affine();
    let r = pallas::Affine::coordinates(&commitment4).unwrap().x().clone();
    let r2 = r.to_repr();
    let mut r3: u128 = 0;
    let mut eight = 1;
    for i in r2{
        r3 += i as u128 * eight;
        eight *= 8;
    }
    
    let selfw_vec = [user1_prikey_share, user2_prikey_share, user3_prikey_share].to_vec();
    let message = generate_random_u128_in_range(1, std::u64::MAX as u128);

    let make_signature = MakeSignature{
        selfk : selfk_vec,
        mta_pub_n :mta_pub_n_vec,
        selfw : selfw_vec,
        pri_p : mta_pri_p_vec,
        pri_q : mta_pri_q_vec,
        message : message,
        r : r3,
    };

    println!("{:?}",r3);

    let sharding_signature = make_signature.make_signature();

    let mut signature = pallas::Scalar::zero();
    for i in sharding_signature{
        signature+=i;
    }

    let k = 17;
    let circuit = MyCircuit {
        input_r : pallas::Scalar::from_u128(r3),
        input_s : signature,
        commitment : commitment4,
        message : pallas::Scalar::from_u128(message),
        pub_key : pub_key,
    };
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()))
    
}
//s = k (m + xr)