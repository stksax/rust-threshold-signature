// use std::ops::{Add, Mul};
use std::fmt::Debug;
use std::ops::Mul;
use std::result;
// use std::result;
use halo2_gadgets::utilities::Var;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::poly::commitment;
use pasta_curves::arithmetic::CurveExt;
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
    let affine_generator = pallas::Affine::generator();
    let kfq = pallas::Scalar::random(rand::rngs::OsRng);
    let k_inv = kfq.invert().unwrap();
    let kep = pallas::Affine::mul(affine_generator, &k_inv);
    let kaf = kep.to_affine();
    let (rx, _, _ ) = kep.jacobian_coordinates();
    let rx1 = rx.to_repr();
    let rx2 = pallas::Scalar::from_repr(rx1).unwrap();
    let pri_key = pallas::Scalar::random(rand::rngs::OsRng);
    let pub_key = pallas::Affine::mul(affine_generator, &pri_key).to_affine();
    let m = pallas::Scalar::random(rand::rngs::OsRng);
    let xr = pallas::Scalar::mul(&pri_key, &rx2);
    let mxr = pallas::Scalar::add(&m, &xr);
    let s = pallas::Scalar::mul(&kfq, &mxr);

    let k = 17;
    let circuit = MyCircuit {
        input_r : rx2,
        input_s : s,
        commitment : kaf,
        message : m,
        pub_key : pub_key,
    };
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    // assert_eq!(prover.verify(), Ok(()))

}
//s = k (m + xr)