//! This module implements various gadgets necessary for folding R1CS types.
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_CHALLENGE_BITS},
  frontend::{num::AllocatedNum, Assignment, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{
      bignat::BigNat,
      util::{f_to_nat, Num},
    },
    utils::{
      alloc_bignat_constant, alloc_one, alloc_scalar_as_base, conditionally_select,
      conditionally_select_bignat, le_bits_to_num,
    },
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait, ROConstantsCircuit},
};
use ff::Field;

/// An Allocated R1CS Instance
#[derive(Clone)]
pub struct AllocatedR1CSInstance<E: Engine> {
  pub(crate) comm_W: Vec<AllocatedPoint<E>>,
  pub(crate) X0: AllocatedNum<E::Base>,
  pub(crate) X1: AllocatedNum<E::Base>,
}

impl<E: Engine> AllocatedR1CSInstance<E> {
  /// Takes the r1cs instance and creates a new allocated r1cs instance
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    u: Option<&R1CSInstance<E>>,
    num_split_wits: usize,
  ) -> Result<Self, SynthesisError> {
    let mut comm_W = Vec::new();

    if u.is_some() {
      for (i, w) in u.as_ref().unwrap().comm_W.iter().enumerate() {
        comm_W.push(AllocatedPoint::alloc(
          cs.namespace(|| format!("allocate W_{}", i)),
          Some(w.to_coordinates()),
        )?);
      }
    } else {
      for i in 0..num_split_wits {
        comm_W.push(AllocatedPoint::alloc(
          cs.namespace(|| format!("allocate W_{}", i)),
          None,
        )?);
      }
    }

    for (i, w) in comm_W.iter().enumerate() {
      w.check_on_curve(cs.namespace(|| format!("check W on curve {}", i)))?;
    }

    let X0 = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate X[0]"), u.map(|u| u.X[0]))?;
    let X1 = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate X[1]"), u.map(|u| u.X[1]))?;

    Ok(AllocatedR1CSInstance { comm_W, X0, X1 })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro(&self, ro: &mut E::ROCircuit) {
    for c in &self.comm_W {
      ro.absorb(&c.x);
      ro.absorb(&c.y);
      ro.absorb(&c.is_infinity);
    }
    ro.absorb(&self.X0);
    ro.absorb(&self.X1);
  }
}

/// An Allocated Relaxed R1CS Instance
pub struct AllocatedRelaxedR1CSInstance<E: Engine> {
  pub(crate) W: Vec<AllocatedPoint<E>>,
  pub(crate) E: AllocatedPoint<E>,
  pub(crate) u: AllocatedNum<E::Base>,
  pub(crate) X0: BigNat<E::Base>,
  pub(crate) X1: BigNat<E::Base>,
}

impl<E: Engine> AllocatedRelaxedR1CSInstance<E> {
  /// Allocates the given `RelaxedR1CSInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    inst: Option<&RelaxedR1CSInstance<E>>,
    num_split_wits: usize,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.

    let mut W = Vec::new();
    if inst.is_some() {
      assert_eq!(num_split_wits, inst.as_ref().unwrap().comm_W.len());
      for (i, w) in inst.as_ref().unwrap().comm_W.iter().enumerate() {
        W.push(AllocatedPoint::alloc(
          cs.namespace(|| format!("allocate W_{}", i)),
          Some(w.to_coordinates()),
        )?);
      }
    } else {
      for i in 0..num_split_wits {
        W.push(AllocatedPoint::alloc(
          cs.namespace(|| format!("allocate W_{}", i)),
          None,
        )?);
      }
    }

    let E = AllocatedPoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    // u << |E::Base| despite the fact that u is a scalar.
    // So we parse all of its bytes as a E::Base element
    let u = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate u"), inst.map(|inst| inst.u))?;

    // Allocate X0 and X1. If the input instance is None, then allocate default values 0.
    let X0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[0]"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let X1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[1]"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[1]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, u = 0, X0 = X1 = 0
  pub fn default<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    num_split_wits: usize,
  ) -> Result<Self, SynthesisError> {
    let mut W = Vec::new();
    for i in 0..num_split_wits {
      W.push(AllocatedPoint::default(
        cs.namespace(|| format!("allocate W_{}", i)),
      )?);
    }
    let E = W[0].clone();

    let u = W[0].x.clone(); // In the default case, W.x = u = 0

    // X0 and X1 are allocated and in the honest prover case set to zero
    // If the prover is malicious, it can set to arbitrary values, but the resulting
    // relaxed R1CS instance with the checked default values of W, E, and u must still be satisfying
    let X0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[0]"),
      || Ok(f_to_nat(&E::Scalar::ZERO)),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let X1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[1]"),
      || Ok(f_to_nat(&E::Scalar::ZERO)),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }

  /// Allocates the R1CS Instance as a `RelaxedR1CSInstance` in the circuit.
  /// E = 0, u = 1
  pub fn from_r1cs_instance<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    inst: AllocatedR1CSInstance<E>,
  ) -> Result<Self, SynthesisError> {
    let E = AllocatedPoint::default(cs.namespace(|| "allocate default E"))?;

    let u = alloc_one(cs.namespace(|| "one"));

    let X0 = BigNat::from_num(
      cs.namespace(|| "allocate X0 from relaxed r1cs"),
      &Num::from(inst.X0),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let X1 = BigNat::from_num(
      cs.namespace(|| "allocate X1 from relaxed r1cs"),
      &Num::from(inst.X1),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    Ok(AllocatedRelaxedR1CSInstance {
      W: inst.comm_W,
      E,
      u,
      X0,
      X1,
    })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    ro: &mut E::ROCircuit,
  ) -> Result<(), SynthesisError> {
    for w in &self.W {
      ro.absorb(&w.x);
      ro.absorb(&w.y);
      ro.absorb(&w.is_infinity);
    }
    ro.absorb(&self.E.x);
    ro.absorb(&self.E.y);
    ro.absorb(&self.E.is_infinity);
    ro.absorb(&self.u);

    // Analyze X0 as limbs
    let X0_bn = self
      .X0
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of X_r[0] to num")))
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // absorb each of the limbs of X[0]
    for limb in X0_bn {
      ro.absorb(&limb);
    }

    // Analyze X1 as limbs
    let X1_bn = self
      .X1
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of X_r[1] to num")))
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // absorb each of the limbs of X[1]
    for limb in X1_bn {
      ro.absorb(&limb);
    }

    Ok(())
  }

  /// Folds self with a relaxed r1cs instance and returns the result
  pub fn fold_with_r1cs<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    params: &AllocatedNum<E::Base>, // hash of R1CSShape of F'
    u: &AllocatedR1CSInstance<E>,
    T: &AllocatedPoint<E>,
    ro_consts: ROConstantsCircuit<E>,
  ) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
    // Compute r:
    let mut ro = E::ROCircuit::new(ro_consts);
    ro.absorb(params);

    // running instance `U` does not need to absorbed since u.X[0] = Hash(params, U, i, z0, zi)
    u.absorb_in_ro(&mut ro);

    ro.absorb(&T.x);
    ro.absorb(&T.y);
    ro.absorb(&T.is_infinity);
    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // W_fold = self.W + r * u.W
    let mut W_fold = Vec::new();
    for (i, (uw, sw)) in u.comm_W.iter().zip(&self.W).enumerate() {
      let rW = uw.scalar_mul(cs.namespace(|| format!("r * u.W {}", i)), &r_bits)?;
      W_fold.push(sw.add(cs.namespace(|| format!("self.W + r * u.W {}", i)), &rW)?);
    }

    // E_fold = self.E + r * T
    let rT = T.scalar_mul(cs.namespace(|| "r * T"), &r_bits)?;
    let E_fold = self.E.add(cs.namespace(|| "self.E + r * T"), &rT)?;

    // u_fold = u_r + r
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*self.u.get_value().get()? + r.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_fold.get_variable() - self.u.get_variable() - r.get_variable(),
    );

    // Fold the IO:
    // Analyze r into limbs
    let r_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      &Num::from(r),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &E::GE::group_params().2,
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Analyze X0 to bignat
    let X0_bn = BigNat::from_num(
      cs.namespace(|| "allocate X0_bn"),
      &Num::from(u.X0.clone()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Fold self.X[0] + r * X[0]
    let (_, r_0) = X0_bn.mult_mod(cs.namespace(|| "r*X[0]"), &r_bn, &m_bn)?;
    // add X_r[0]
    let r_new_0 = self.X0.add(&r_0)?;
    // Now reduce
    let X0_fold = r_new_0.red_mod(cs.namespace(|| "reduce folded X[0]"), &m_bn)?;

    // Analyze X1 to bignat
    let X1_bn = BigNat::from_num(
      cs.namespace(|| "allocate X1_bn"),
      &Num::from(u.X1.clone()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Fold self.X[1] + r * X[1]
    let (_, r_1) = X1_bn.mult_mod(cs.namespace(|| "r*X[1]"), &r_bn, &m_bn)?;
    // add X_r[1]
    let r_new_1 = self.X1.add(&r_1)?;
    // Now reduce
    let X1_fold = r_new_1.red_mod(cs.namespace(|| "reduce folded X[1]"), &m_bn)?;

    Ok(Self {
      W: W_fold,
      E: E_fold,
      u: u_fold,
      X0: X0_fold,
      X1: X1_fold,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    other: &AllocatedRelaxedR1CSInstance<E>,
    condition: &Boolean,
  ) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
    let mut W = Vec::new();
    for (i, (sw, ow)) in self.W.iter().zip(&other.W).enumerate() {
      W.push(AllocatedPoint::conditionally_select(
        cs.namespace(|| format!("W = cond ? self.W : other {}", i)),
        &sw,
        &ow,
        condition,
      )?);
    }

    let E = AllocatedPoint::conditionally_select(
      cs.namespace(|| "E = cond ? self.E : other.E"),
      &self.E,
      &other.E,
      condition,
    )?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let X0 = conditionally_select_bignat(
      cs.namespace(|| "X[0] = cond ? self.X[0] : other.X[0]"),
      &self.X0,
      &other.X0,
      condition,
    )?;

    let X1 = conditionally_select_bignat(
      cs.namespace(|| "X[1] = cond ? self.X[1] : other.X[1]"),
      &self.X1,
      &other.X1,
      condition,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }
}
