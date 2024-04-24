use std::fmt::Debug;
use std::ops::{Add, Mul};
use ff::Field;
use pasta_curves::arithmetic::CurveExt;
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
struct MyCircuit {
    input_r : pallas::Scalar,
    input_s : pallas::Scalar,
    commitment : pallas::Affine,
    message : pallas::Scalar,
    pub_key :  pallas::Affine,
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
        let affine_generator = pallas::Affine::generator();

        let s_inv = pallas::Scalar::invert(&self.input_s).unwrap();
        let s_m_inv = pallas::Scalar::mul(&s_inv, &self.message);
        let scalar1 = pallas::Affine::mul(affine_generator, s_m_inv).to_affine();
        
        let p1 = Point::new(
            chip.clone(), 
            layouter.namespace(|| "(s⁻1 * M ) * G"),
            Value::known(scalar1),
        )?;
    
        let pub_key = NonIdentityPoint::new(
            chip.clone(), 
            layouter.namespace(|| "pub_key"),
            Value::known(self.pub_key),
        )?;

        let commitment =  Point::new(
            chip.clone(), 
            layouter.namespace(|| "K⁻1 * G"),
            Value::known(self.commitment),
        )?;
     
        let s_r_inv = pallas::Scalar::mul(&s_inv, &self.input_r);
        let s_r_inv_fp = pallas::Base::from_repr(s_r_inv.to_repr()).unwrap();
        let base = chip.load_private(
            layouter.namespace(|| "S⁻1 * r"), 
            column, 
            Value::known(s_r_inv_fp),
        )?;
        let scalar2 = ScalarVar::from_base(
            chip.clone(), 
            layouter.namespace(|| "S⁻1 * r"), 
            &base,
        )?;

        let (p2,_) = NonIdentityPoint::mul(
            &pub_key, 
            layouter.namespace(|| "(S⁻1 * r )* G"), 
            scalar2,
        )?;

        let p3 = Point::add(
            &p1, 
            layouter.namespace(|| "(s⁻1 * M ) * G + (S⁻1 * r )* G"), 
            &p2,
        )?;

        let result = Point::constrain_equal(
            &commitment, 
            layouter.namespace(|| "(s⁻1 * M ) * G + (S⁻1 * r )* G = K⁻1 * G"), 
            &p3,
        );

        result
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
fn test_ecdsa() {
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
    assert_eq!(prover.verify(), Ok(()))
}
}