//! There are two augmented circuits: the primary and the secondary.
//! Each of them is over a curve in a 2-cycle of elliptic curves.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is H(params = H(shape, ck), i, z0, zi, U).
//! Each circuit folds the last invocation of the other into the running instance

use crate::{
  constants::NUM_HASH_BITS,
  frontend::{
    num::AllocatedNum, AllocatedBit, Assignment, Boolean, ConstraintSystem, SynthesisError,
  },
  gadgets::{
    ecc::AllocatedPoint,
    utils::{
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec, le_bits_to_num,
    },
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Engine, ROCircuitTrait, ROConstantsCircuit,
  },
  Commitment,
};
use ff::Field;
use serde::{Deserialize, Serialize};

mod r1cs;
use r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance};

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NovaAugmentedCircuitInputs<E: Engine> {
  pp_digest: E::Scalar,
  i: E::Base,
  z0: Vec<E::Base>,
  zi: Option<Vec<E::Base>>,
  ram_hints: Option<Vec<E::Base>>,
  Ci: Option<Vec<E::Base>>,
  U: Option<RelaxedR1CSInstance<E>>,
  ri: Option<E::Base>,
  r_next: E::Base,
  u: Option<R1CSInstance<E>>,
  T: Option<Commitment<E>>,
}

impl<E: Engine> NovaAugmentedCircuitInputs<E> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    pp_digest: E::Scalar,
    i: E::Base,
    z0: Vec<E::Base>,
    zi: Option<Vec<E::Base>>,
    ram_hints: Option<Vec<E::Base>>,
    Ci: Option<Vec<E::Base>>,
    U: Option<RelaxedR1CSInstance<E>>,
    ri: Option<E::Base>,
    r_next: E::Base,
    u: Option<R1CSInstance<E>>,
    T: Option<Commitment<E>>,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
      ram_hints,
      Ci,
      U,
      ri,
      r_next,
      u,
      T,
    }
  }
}

/// The augmented circuit F' in Nova that includes a step circuit F
/// and the circuit for the verifier in Nova's non-interactive folding scheme
pub struct NovaAugmentedCircuit<'a, E: Engine, SC: StepCircuit<E::Base>> {
  is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
  ro_consts: ROConstantsCircuit<E>,
  inputs: Option<NovaAugmentedCircuitInputs<E>>,
  step_circuit: &'a mut SC,   // The function that is applied for each step
  num_split_witnesses: usize, // num of cmts in the OPPOSITE circuit
  ram_batch_sizes: Vec<usize>,
}

impl<'a, E: Engine, SC: StepCircuit<E::Base>> NovaAugmentedCircuit<'a, E, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    is_primary_circuit: bool,
    inputs: Option<NovaAugmentedCircuitInputs<E>>,
    step_circuit: &'a mut SC,
    ro_consts: ROConstantsCircuit<E>,
    num_split_witnesses: usize,
    ram_batch_sizes: Vec<usize>,
  ) -> Self {
    Self {
      is_primary_circuit,
      inputs,
      step_circuit,
      ro_consts,
      num_split_witnesses,
      ram_batch_sizes,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
    total_ram: usize,
  ) -> Result<
    (
      AllocatedNum<E::Base>,
      AllocatedNum<E::Base>,
      Vec<AllocatedNum<E::Base>>,
      Vec<AllocatedNum<E::Base>>,
      Vec<AllocatedNum<E::Base>>,
      Option<Vec<AllocatedNum<E::Base>>>,
      AllocatedRelaxedR1CSInstance<E>,
      AllocatedNum<E::Base>,
      AllocatedNum<E::Base>,
      AllocatedR1CSInstance<E>,
      AllocatedPoint<E>,
    ),
    SynthesisError,
  > {
    // Allocate ram cmt wits (from z_i+1, supplied here with some prover help)
    let zero = vec![E::Base::ZERO; total_ram];
    let ram_hints = (0..total_ram)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("ram_hints_{i}")), || {
          Ok(self.inputs.get()?.ram_hints.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // Allocate pp_digest
    let pp_digest = alloc_scalar_as_base::<E, _>(
      cs.namespace(|| "pp_digest"),
      self.inputs.as_ref().map(|inputs| inputs.pp_digest),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // Allocated Ci (only if split)
    let zero = vec![E::Base::ZERO; self.num_split_witnesses - 1];
    let C_i = if self.ram_batch_sizes.len() > 0 {
      Some(
        (0..(self.num_split_witnesses - 1))
          .map(|i| {
            AllocatedNum::alloc(cs.namespace(|| format!("Ci_{i}")), || {
              Ok(self.inputs.get()?.Ci.as_ref().unwrap_or(&zero)[i])
            })
          })
          .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?,
      )
    } else {
      None
    };

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![E::Base::ZERO; arity];

    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // Allocate the running instance
    let U: AllocatedRelaxedR1CSInstance<E> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
      self.num_split_witnesses,
    )?;

    // Allocate ri
    let r_i = AllocatedNum::alloc(cs.namespace(|| "ri"), || {
      Ok(self.inputs.get()?.ri.unwrap_or(E::Base::ZERO))
    })?;

    // Allocate r_i+1
    let r_next = AllocatedNum::alloc(cs.namespace(|| "r_i+1"), || Ok(self.inputs.get()?.r_next))?;

    // Allocate the instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.as_ref().and_then(|inputs| inputs.u.as_ref()),
      self.num_split_witnesses,
    )?;

    // Allocate T
    let T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.T.map(|T| T.to_coordinates())),
    )?;
    T.check_on_curve(cs.namespace(|| "check T on curve"))?;

    Ok((pp_digest, i, z_0, z_i, ram_hints, C_i, U, r_i, r_next, u, T))
  }

  fn synthesize_hash_check<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Base>,
    i: &AllocatedNum<E::Base>,
    z_0: &[AllocatedNum<E::Base>],
    z_i: &[AllocatedNum<E::Base>],
    C_next: &Option<Vec<AllocatedNum<E::Base>>>,
    U: &AllocatedRelaxedR1CSInstance<E>,
    r_i: &AllocatedNum<E::Base>,
  ) -> Result<AllocatedNum<E::Base>, SynthesisError> {
    // Check that u.x[0] = Hash(pp_digest, i, z_0, z_i, U, r_i)
    let mut ro = E::ROCircuit::new(self.ro_consts.clone());
    ro.absorb(pp_digest);
    ro.absorb(i);
    for e in z_0 {
      ro.absorb(e);
    }
    for e in z_i {
      ro.absorb(e);
    }
    if self.ram_batch_sizes.len() > 0 {
      for c in C_next.as_ref().unwrap() {
        ro.absorb(c);
      }
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    ro.absorb(r_i);

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;

    Ok(hash)
  }

  /// Synthesizes base case and returns the new relaxed `R1CSInstance`
  fn synthesize_base_case<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstance<E>,
  ) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
    let U_default: AllocatedRelaxedR1CSInstance<E> = if self.is_primary_circuit {
      // The primary circuit just returns the default R1CS instance
      AllocatedRelaxedR1CSInstance::default(
        cs.namespace(|| "Allocate U_default"),
        self.num_split_witnesses,
      )?
    } else {
      // The secondary circuit returns the incoming R1CS instance
      AllocatedRelaxedR1CSInstance::from_r1cs_instance(cs.namespace(|| "Allocate U_default"), u)?
    };
    Ok(U_default)
  }

  /// Synthesizes non base case and returns the new relaxed `R1CSInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Base>,
    C_i: &Option<Vec<AllocatedNum<E::Base>>>,
    U: &AllocatedRelaxedR1CSInstance<E>,
    r_i: &AllocatedNum<E::Base>,
    u: &AllocatedR1CSInstance<E>,
    T: &AllocatedPoint<E>,
  ) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      pp_digest,
      u,
      T,
      self.ro_consts.clone(),
    )?;

    Ok(U_fold)
  }
}

impl<E: Engine, SC: StepCircuit<E::Base>> NovaAugmentedCircuit<'_, E, SC> {
  /// synthesize circuit giving constraint system
  pub fn synthesize<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut self,
    cs: &mut CS,
  ) -> Result<
    (
      Vec<AllocatedNum<E::Base>>,
      Option<Vec<AllocatedNum<E::Base>>>,
    ),
    SynthesisError,
  > {
    let total_ram = self.ram_batch_sizes.iter().sum();
    let arity = self.step_circuit.arity() - total_ram;

    // Allocate all witnesses
    let (pp_digest, i, z_0, z_i, ram_hints, C_i, U, r_i, r_next, u, T) = self.alloc_witness(
      cs.namespace(|| "allocate the circuit witness"),
      arity,
      total_ram,
    )?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // compute hash of the non-deterministic inputs
    let hash = self.synthesize_hash_check(
      cs.namespace(|| "synthesize input hash check"),
      &pp_digest,
      &i,
      &z_0,
      &z_i,
      &C_i,
      &U,
      &r_i,
    )?;

    let check_non_base_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    // Synthesize the circuit for the base case and get the new running instance
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "base case"), u.clone())?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let Unew_non_base = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &pp_digest,
      &C_i,
      &U,
      &r_i,
      &u,
      &T,
    )?;

    // Either check_non_base_pass=true or we are in the base case
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      &check_non_base_pass,
      &is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    // Compute the U_new
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      &Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E::Base::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;

    // ram coming in is never looked at - no need to hook up
    let mut z_input_full = (0..total_ram)
      .map(|i| AllocatedNum::alloc(cs.namespace(|| format!("ram_in_{i}")), || Ok(E::Base::ZERO)))
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;
    z_input_full.extend(z_input);

    let z_next_full = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input_full)?;

    let ram_from_z = &z_next_full[0..total_ram];
    let z_next = &z_next_full[total_ram..];

    // make sure ram hints are tied in // TODO for mult cmts
    for j in 0..ram_hints.len() {
      cs.enforce(
        || format!("ram hint {}", j),
        |lc| lc + ram_hints[j].get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + ram_from_z[j].get_variable(),
      );
    }

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // Accumulate Commitments
    let C_next = if self.ram_batch_sizes.len() > 0 {
      let mut cmts = Vec::new();
      // assert!(!self.params.is_primary_circuit);
      assert!(u.comm_W.len() > 1);

      for wi in 0..(u.comm_W.len() - 1) {
        let mut cc = E::ROCircuit::new(
          self.ro_consts.clone(), // TODO?
        );

        cc.absorb(&C_i.as_ref().unwrap()[wi]);

        cc.absorb(&u.comm_W[wi].x);
        cc.absorb(&u.comm_W[wi].y);
        cc.absorb(&u.comm_W[wi].is_infinity);

        let cc_hash_bits = cc.squeeze(cs.namespace(|| "cc output hash bits"), NUM_HASH_BITS)?;
        let hash_num = le_bits_to_num(cs.namespace(|| "cc convert hash to num"), &cc_hash_bits)?;

        cmts.push(hash_num);
      }
      Some(cmts)
    } else {
      None
    };

    // Compute the new hash H(pp_digest, Unew, i+1, z0, z_{i+1})
    let hash = self.synthesize_hash_check(
      cs.namespace(|| "synthesize output hash check"),
      &pp_digest,
      &i_new,
      &z_0,
      &z_next,
      &C_next,
      &Unew,
      &r_next,
    )?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    u.X1
      .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok((z_next.to_vec(), C_next))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    gadgets::utils::scalar_as_base,
    provider::{
      Bn256EngineKZG, GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{circuit::TrivialCircuit, snark::default_ck_hint},
  };

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_recursive_circuit_with<E1, E2>(
    num_constraints_primary: usize,
    num_constraints_secondary: usize,
  ) where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let ro_consts1 = ROConstantsCircuit::<E2>::default();
    let ro_consts2 = ROConstantsCircuit::<E1>::default();

    let mut tc1 = TrivialCircuit::default();
    // Initialize the shape and ck for the primary
    let circuit1: NovaAugmentedCircuit<'_, E2, TrivialCircuit<<E2 as Engine>::Base>> =
      NovaAugmentedCircuit::new(true, None, &mut tc1, ro_consts1.clone(), 1, vec![]);
    let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, ck1) = cs.r1cs_shape(&*default_ck_hint(), vec![]);
    assert_eq!(cs.num_constraints(), num_constraints_primary);

    let mut tc2 = TrivialCircuit::default();
    // Initialize the shape and ck for the secondary
    let circuit2: NovaAugmentedCircuit<'_, E1, TrivialCircuit<<E1 as Engine>::Base>> =
      NovaAugmentedCircuit::new(false, None, &mut tc2, ro_consts2.clone(), 1, vec![]);
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, ck2) = cs.r1cs_shape(&*default_ck_hint(), vec![]);
    assert_eq!(cs.num_constraints(), num_constraints_secondary);

    // Execute the base case for the primary
    let zero1 = <<E2 as Engine>::Base as Field>::ZERO;
    let ri_1 = <<E2 as Engine>::Base as Field>::ZERO;
    let mut cs1 = SatisfyingAssignment::<E1>::new();
    let inputs1: NovaAugmentedCircuitInputs<E2> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(zero1), // pass zero for testing
      zero1,
      vec![zero1],
      None,
      None,
      None,
      None,
      None,
      ri_1,
      None,
      None,
    );
    let circuit1: NovaAugmentedCircuit<'_, E2, TrivialCircuit<<E2 as Engine>::Base>> =
      NovaAugmentedCircuit::new(true, Some(inputs1), &mut tc1, ro_consts1, 1, vec![]);
    let _ = circuit1.synthesize(&mut cs1);
    let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &ck1, None).unwrap();
    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&ck1, &inst1, &witness1).is_ok());

    // Execute the base case for the secondary
    let zero2 = <<E1 as Engine>::Base as Field>::ZERO;
    let ri_2 = <<E1 as Engine>::Base as Field>::ZERO;
    let mut cs2 = SatisfyingAssignment::<E2>::new();
    let inputs2: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E2>(zero2), // pass zero for testing
      zero2,
      vec![zero2],
      None,
      None,
      None,
      None,
      None,
      ri_2,
      Some(inst1),
      None,
    );
    let circuit2: NovaAugmentedCircuit<'_, E1, TrivialCircuit<<E1 as Engine>::Base>> =
      NovaAugmentedCircuit::new(false, Some(inputs2), &mut tc2, ro_consts2, 1, vec![]);
    let _ = circuit2.synthesize(&mut cs2);
    let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &ck2, None).unwrap();
    // Make sure that it is satisfiable
    assert!(shape2.is_sat(&ck2, &inst2, &witness2).is_ok());
  }

  #[test]
  fn test_recursive_circuit() {
    test_recursive_circuit_with::<PallasEngine, VestaEngine>(9817, 10349);
    test_recursive_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(9985, 10538);
    test_recursive_circuit_with::<Secp256k1Engine, Secq256k1Engine>(10264, 10961);
  }
}
