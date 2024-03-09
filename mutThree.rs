
// use std::ops::{Add, Mul};

use std::fmt::Debug;
use std::result;
// use std::result;

use halo2_proofs::arithmetic::Field;

use pasta_curves::group::cofactor::CofactorCurveAffine;
use pasta_curves::group::ff::{FromUniformBytes, PrimeField};
use pasta_curves::group::{Curve, Group, GroupEncoding};

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value, Chip},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2_gadgets::utilities::{lookup_range_check::LookupRangeCheckConfig, UtilitiesInstructions};
use halo2_gadgets::ecc::*;
// fn print_type_of<T>(_: &T) {

//     println!("{}", std::any::type_name::<T>())
    
//     } 

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
#[derive(Default)]
struct MyCircuit {
    input_r : u128,
    input_s : u128,
    msg: u128,
    pub_key : pallas::Affine,
    input_k : u128,
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

        // let msg = pallas::Base::from_raw([
        //         0x4d90ab820b12320a,
        //         0xd976bbfabbc5661d,
        //         0x573b3d7f7d681310,
        //         0x17033d3c60c68173,
        //     ]);
        
        let pub_key = NonIdentityPoint::new(
            chip.clone(),
            layouter.namespace(|| "pub_key"),
            Value::known(self.pub_key),
        )?;
        
        let s = pallas::Scalar::from_u128(self.input_s);
        let r = pallas::Scalar::from_u128(self.input_r);
        let k = pallas::Scalar::from_u128(self.input_k);
        let s_inv = pallas::Scalar::invert(&s).unwrap();
        let m = pallas::Scalar::from_u128(self.msg);
        let u1 = pallas::Scalar::mul(&s_inv, &m);
        let u2 = pallas::Scalar::mul(&s_inv, &r);

        let generatorPointAffine = pallas::Affine::generator();
        let generatorPoint = NonIdentityPoint::new(
            chip.clone(),
            layouter.namespace(|| "gen"),
            Value::known(generatorPointAffine),
        )?;

        let scalar1 = ScalarFixed::new(
            chip.clone(), 
            layouter.namespace(|| "scalar1"),
            Value::known(pallas::Scalar::mul(&s, &r)),
        )?;

        let fixPoint = FullWidth::from_pallas_generator();

        let fixPointBase = EccFixedPoint::from_inner(
            chip.clone(), 
            fixPoint,
        );
 
        let (p1, _) = EccFixedPoint::mul(
            &fixPointBase, 
            layouter.namespace(|| "pub_key"),
            scalar1,
        )?;
        
        // let scalar2 = ScalarVar::new(
        //     chip.clone(), 
        //     layouter.namespace(|| "pub_key"),
        //     Value::known(u2),
        // )?;

        let fixBaseField =FixedPointBaseField::from_inner(
            chip.clone(),
            BaseField, 
        );
        
        let fps = pallas::Base::from_u128(self.input_s);
        let fpr = pallas::Base::from_u128(self.input_r);
        let scalar_fixed = chip.load_private(
            layouter.namespace(|| "random base field element"),
            column,
            Value::known(pallas::Base::mul(&fps, &fpr)),
        )?;

        let p2 = FixedPointBaseField::mul(
            &fixBaseField, 
            layouter.namespace(|| "pub_key"),
            scalar_fixed,
        )?;

        let v1  = chip.load_private(
            layouter.namespace(|| "random base field element"),
            column,
            Value::known(pallas::Base::from_u128(self.input_s)),
        )?;
        let v2  = chip.load_private(
            layouter.namespace(|| "random base field element"),
            column,
            Value::known(pallas::Base::from_u128(self.input_r)),
        )?;
        let scalar3 = ScalarFixedShort::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key"),
            (v1, v2),
        )?;


        let shortPoint = FixedPointShort::from_inner(
            chip.clone(), 
            Short,
        );
       
        let (p3, _) = FixedPointShort::mul(
            &shortPoint, 
            layouter.namespace(|| "pub_key"),
            scalar3,
        )?;

        let result = Point::constrain_equal(
            &p1, 
            layouter.namespace(|| "pub_key"),
            &p2,
        );

        println!("{:?}",result);

        let u1 = Point::extract_p(&p1);
        let u2 = Point::extract_p(&p2);
        let u3 = Point::extract_p(&p3);
        println!("{:#?}",&u1);
        println!("{:#?}",&u2);
        println!("{:#?}",&u3);

        result
    }
    
}

fn main() {
    // let message = pallas::Base::from_raw([
    //     0x4d90ab820b12320a,
    //     0xd976bbfabbc5661d,
    //     0x573b3d7f7d681310,
    //     0x17033d3c60c68173,
    // ]);

    let pubKey = pallas::Point::random(rand::rngs::OsRng).to_affine();

    let k = 17;
    let circuit = MyCircuit {
        input_r : 457824567,
        input_s : 12345678765,
        msg : 7654345678,
        pub_key : pubKey,
        input_k : 567865432345,
    };
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    // assert_eq!(prover.verify(), Ok(()))
    

}
// let m = Value::known(&self.msg);
//         let xr = Value::mul(r3, Value::known(pri_key_fp));
//         let mxr = Value::add(xr, m);
//         let s = Value::mul(mxr, Value::known(k));
//         let s2 = Value::to_field(&s);
//         let s_inv = Value::invert(&s2);
//         let s_inv2 = Value::evaluate(s_inv);
//         let sm_inv = Value::mul(m, s_inv2);
        
//         let scalar2 = chip.load_private(
//             layouter.namespace(|| "random7 scalar1"),
//             column,
//             sm_inv,
//         )?;
//         let u1_scalar = ScalarVar::from_base(
//             chip.clone(),
//             layouter.namespace(|| "ScalarVar12 s_inv*m1"),
//             &scalar2,
//             )?;
//         let (u1, _) = NonIdentityPoint::mul(
//             &base_point,
//             layouter.namespace(|| "random12 scalar2"),
//             u1_scalar,
//         )?;
//         let sr_inv = Value::mul(r3, s_inv2);
//         let scalar3 = chip.load_private(
//             layouter.namespace(|| "random 33scalar3"),
//             column,
//             sr_inv,
//         )?;
//         let u2_scalar = ScalarVar::from_base(
//             chip.clone(),
//             layouter.namespace(|| "ScalarVar45 s_inv*m4"),
//             &scalar3,
//             )?;
//         let (u2, _) = NonIdentityPoint::mul(
//             &pub_key,
//             layouter.namespace(|| "random 23scalar5"),
//             u2_scalar,
//         )?;

//         let u3 = Point::add(
//             &u1, 
//             layouter.namespace(|| "random 12scalar12")
//             , &u2
//         )?;

//         let result: Result<(), Error> = Point::constrain_equal(&u3, 
//             layouter.namespace(|| "rando s3calar13"), 
//             &commitment,
//         );
//         let u33 = Point::extract_p(&u3);
//         let u34 = Point::extract_p(&commitment);
//         // let u35 = X::inner(&u34);
//         // let u36 = u35.value().cloned();

//         let check = Value::mul(s, Value::known(k_inv));
  
//         println!("{:?}",&u33);
//         println!("------------------");
//         println!("{:?}",&u34);
//         println!("------------------");
//         println!("{:?}",&check);
//         println!("{:?}",&mxr);

//         //test
//         let t12 = pallas::Base::from_u128(12);
//         let t1 = pallas::Base::neg(&t12);
//         let t2 = pallas::Base::from_u128(12);
//         let t3 = pallas::Base::from_u128(20);
//         let scalart1 = chip.load_private(
//             layouter.namespace(|| "random1 scalar"),
//             column,
//             Value::known(t1),
//         )?;

//         let t1_scalar = ScalarVar::from_base(
//         chip.clone(),
//         layouter.namespace(|| "ScalarVar2 from_k_inv"),
//         &scalart1,
//         )?;

//         let (t1p, _) = NonIdentityPoint::mul(
//             &base_point ,
//             layouter.namespace(|| "random [K_inv]G"),
//             t1_scalar)?;
        
//             let scalart2 = chip.load_private(
//                 layouter.namespace(|| "random1 scalar"),
//                 column,
//                 Value::known(t2),
//             )?;
    
//             let t2_scalar = ScalarVar::from_base(
//             chip.clone(),
//             layouter.namespace(|| "ScalarVar2 from_k_inv"),
//             &scalart2,
//             )?;
    
//             let (t2p, _) = NonIdentityPoint::mul(
//                 &base_point ,
//                 layouter.namespace(|| "random [K_inv]G"),
//                 t2_scalar)?;
        
//                 let scalart3 = chip.load_private(
//                     layouter.namespace(|| "random1 scalar"),
//                     column,
//                     Value::known(t3),
//                 )?;
        
//                 let t3_scalar = ScalarVar::from_base(
//                 chip.clone(),
//                 layouter.namespace(|| "ScalarVar2 from_k_inv"),
//                 &scalart3,
//                 )?;
        
//                 let (t3p, _) = NonIdentityPoint::mul(
//                     &base_point ,
//                     layouter.namespace(|| "random [K_inv]G"),
//                     t3_scalar)?;
//         let t4p = Point::add(&t1p, layouter, &t2p).unwrap();
//         let t3px = Point::extract_p(&t3p);
//         let t4px = Point::extract_p(&t4p);
//         println!("{:?}",&t3px);
//         println!("{:?}",&t4px);
//         println!("{:?}",&t1);
// let column = chip.config().advices[0];
        
// let k = pallas::Base::random(rand::rngs::OsRng);
// let k_inv = pallas::Base::invert(&k).unwrap();

// let scalar1 = chip.load_private(
//     layouter.namespace(|| "random1 scalar"),
//     column,
//     Value::known(k_inv),
// )?;

// let k_inv_scalar = ScalarVar::from_base(
// chip.clone(),
// layouter.namespace(|| "ScalarVar2 from_k_inv"),
// &scalar1,
// )?;

// let base = pallas::Point::generator().to_affine();
// let base_point = NonIdentityPoint::new(
//     chip.clone(),
//     layouter.namespace(|| "G"),
//     Value::known(base),
// )?;

// let (commitment, _) = NonIdentityPoint::mul(
//     &base_point ,
//     layouter.namespace(|| "random [K_inv]G"),
//     k_inv_scalar)?;

// let r = pallas::Scalar::from_repr(self.input_r).unwrap();
// let s = pallas::Scalar::from_repr(self.input_s).unwrap();
// let s_inv = pallas::Scalar::invert(&s).unwrap();
// let m = pallas::Scalar::from_repr(self.msg).unwrap();




// let mut random_num = rand::rngs::OsRng;
// let ii:u64 = OsRng::next_u64( &mut random_num);
// let pri_key_fp = pallas::Base::from(ii);

// let pri_key_fq = pallas::Scalar::from(ii);
// let one = pallas::Point::generator();
// let p = pallas::Point::mul(one, pri_key_fq);// waire
// let pri_key_affine = pallas::Affine::from(p);
// let scalar111 = chip.load_private(
//     layouter.namespace(|| "random3 scalar"),
//     column,
//     Value::known(pri_key_fp),
// )?;
// let scalar222 =  ScalarVar::from_base(
//     chip.clone(),
//     layouter.namespace(|| "ScalarVar4 s_inv*m"),
//     &scalar111,
//     )?;
// let (pk2, _) = NonIdentityPoint::mul(
//     &base_point,
//     layouter.namespace(|| "random5 scalar"),
//     scalar222,
// )?;

// let pub_key = NonIdentityPoint::new(
//     chip.clone(),
//     layouter.namespace(|| "random6 [K_inv]G"),
//     Value::known(pri_key_affine),
// )?;