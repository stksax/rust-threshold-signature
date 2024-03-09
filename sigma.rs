
// use std::ops::{Add, Mul};

use std::fmt::Debug;
use std::ops::Mul;
use std::result;
// use std::result;

use halo2_proofs::arithmetic::Field;

use halo2_proofs::poly::commitment;
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
    input_r : pallas::Scalar,
    pub_key : pallas::Affine,
    input_e : u128,
    commitment : pallas::Affine,
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
        
        let one = pallas::Affine::generator();
        let scalac1 = pallas::Affine::mul(one, self.input_r).to_affine();
        let pubKey = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "d"), 
            Value::known(self.pub_key),
        )?;

        let p1 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "d"), 
            Value::known(self.commitment),
        )?;

        // let p1 = Point::new(
        //     chip.clone(), 
        //     layouter.namespace(|| "d"), 
        //     Value::known(scalac1),
        // )?;

        let var = chip.load_private(
            layouter.namespace(|| "d"), 
            column, 
            Value::known(pallas::Base::from_u128(self.input_e))
        )?;

        let scalar2 = ScalarVar::from_base(
            chip.clone(), 
            layouter.namespace(|| "d"), 
            &var,
        )?;

        let (p2, _) = NonIdentityPoint::mul(
            &pubKey, 
            layouter.namespace(|| "d"), 
            scalar2,
        )?;
        
        let p3 = Point::add(
            &p1, 
            layouter.namespace(|| "pub_key"), 
            &p2,
        )?;

        let p4 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key"), 
            Value::known(pallas::Affine::mul(one, self.input_r).to_affine()),
        )?;
      
        let result = Point::constrain_equal(
            &p3, 
            layouter.namespace(|| "pub_key"),
            &p4,
        );

        println!("{:?}",result);

        let u1 = Point::extract_p(&p3);
        let u2 = Point::extract_p(&p4);
       
        println!("{:?}",&u1);
        println!("{:?}",&u2);

        result
    }
    
}

fn main() {
    let k1 = 457824567;
    let k = pallas::Scalar::from_u128(k1);
    let e = 567865432345;
    // let e = pallas::Scalar::from_u128(e1);

    let prikey = 46578;
    let one = pallas::Affine::generator();
    let prifq = pallas::Scalar::from_u128(prikey);
    let pubKey = pallas::Affine::mul(one, &prifq).to_affine();
    let ep = pallas::Scalar::mul(&prifq, &pallas::Scalar::from_u128(e));
    let r = pallas::Scalar::add(&k, &ep);
    let commitment = pallas::Affine::mul(one, &k).to_affine();



    let k = 17;
    let circuit = MyCircuit {
        input_r : r,
        pub_key : pubKey,
        input_e : e,
        commitment : commitment,
    };
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    // assert_eq!(prover.verify(), Ok(()))

}
//r = k + e*pri