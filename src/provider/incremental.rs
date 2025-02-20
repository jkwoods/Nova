//! This module provides an incremental commitment scheme
use crate::{
  provider::{poseidon::PoseidonConstantsCircuit, traits::DlogGroup, PoseidonRO},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, GetGeneratorsTrait},
    ROCircuitTrait, ROTrait,
  },
  CommitmentKey, Engine, ROConstants, NUM_HASH_BITS,
};
use ff::Field;
use rand_core::OsRng;
use std::marker::PhantomData;

/// Incremental Commitment Scheme Generators
pub struct Incremental<E1: Engine, E2: Engine>
where
  E1::GE: DlogGroup,
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  ped_gen: CommitmentKey<E1>,
  pos_constants: ROConstants<E1>,
  _p: PhantomData<E2>,
}

impl<E1: Engine, E2: Engine> Incremental<E1, E2>
where
  E1::GE: DlogGroup,
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  CommitmentKey<E1>: GetGeneratorsTrait<E1>,
{
  /// setup generators
  pub fn setup(ped_gen: CommitmentKey<E1>, batch_size: usize) -> Self {
    assert_eq!(ped_gen.get_ck().len(), batch_size);

    // ROCC<E1> (to match secondary)
    let pos_constants: ROConstants<E1> = ROConstants::<E1>::default();

    Incremental::<E1, E2> {
      ped_gen,
      pos_constants,
      _p: PhantomData::default(),
    }
  }

  // TODO: iron out return type
  /// commit incrementally to chunk of list
  pub fn commit(&self, c_i: Option<E2::Scalar>, w: &[E1::Scalar]) -> (E1::Scalar, E1::Scalar) {
    let mut cc = E1::RO::new(self.pos_constants.clone(), 4);

    if c_i.is_none() {
      cc.absorb(E2::Scalar::ZERO);
    } else {
      cc.absorb(c_i.unwrap());
    }

    let blind = E1::Scalar::random(&mut OsRng);
    let ped_cmt = E1::CE::commit(&self.ped_gen, w, &blind);
    let ped_coords = ped_cmt.to_coordinates();

    cc.absorb(ped_coords.0);
    cc.absorb(ped_coords.1);
    cc.absorb(if ped_coords.2 {
      E2::Scalar::ONE
    } else {
      E2::Scalar::ZERO
    });

    (cc.squeeze(NUM_HASH_BITS), blind)
  }
}
