//! This module provides an incremental commitment scheme
use crate::{
  provider::{
    pedersen::CommitmentKeyExtTrait, poseidon::PoseidonConstantsCircuit, traits::DlogGroup,
    PoseidonRO,
  },
  scalar_as_base,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, GetGeneratorsTrait},
    ROCircuitTrait, ROTrait,
  },
  Commitment, CommitmentKey, Engine, ROConstants, NUM_HASH_BITS,
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
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  /// pedersen gens  
  pub ped_gen: CommitmentKey<E1>,
  pos_constants: ROConstants<E1>,
  _p: PhantomData<E2>,
}

impl<E1: Engine, E2: Engine> Incremental<E1, E2>
where
  E1::GE: DlogGroup,
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  CommitmentKey<E1>: CommitmentKeyExtTrait<E1>,
{
  /// setup generators
  pub fn setup(label: &'static [u8], size: usize) -> Self {
    let ped_gen = E1::CE::setup(label, size);
    let pos_constants: ROConstants<E1> = ROConstants::<E1>::default();

    Incremental::<E1, E2> {
      ped_gen,
      pos_constants,
      _p: PhantomData::default(),
    }
  }

  /// split generators for split wits
  pub fn split_at(&self, n: usize) -> (Self, Self) {
    let (ped_gen_a, ped_gen_b) = self.ped_gen.split_at(n);

    (
      Incremental::<E1, E2> {
        ped_gen: ped_gen_a,
        pos_constants: self.pos_constants.clone(),
        _p: PhantomData::default(),
      },
      Incremental::<E1, E2> {
        ped_gen: ped_gen_b,
        pos_constants: self.pos_constants.clone(),
        _p: PhantomData::default(),
      },
    )
  }

  /// commit incrementally to chunk of list
  pub fn commit(&self, c_i: Option<E2::Scalar>, w: &[E1::Scalar]) -> (E2::Scalar, E1::Scalar) {
    let mut cc = E1::RO::new(self.pos_constants.clone(), 4);

    if c_i.is_none() {
      cc.absorb(E2::Scalar::ZERO);
    } else {
      cc.absorb(c_i.unwrap());
    }

    let blind = E1::Scalar::random(&mut OsRng);
    println!("committing to {:#?} with blind {:#?}", w, blind);
    let ped_cmt = E1::CE::commit(&self.ped_gen, w, &blind);

    println!("cmt in clear {:#?}", ped_cmt);

    let ped_coords = ped_cmt.to_coordinates();
    println!("x {:#?}", ped_coords.0);

    cc.absorb(ped_coords.0);
    cc.absorb(ped_coords.1);
    cc.absorb(if ped_coords.2 {
      E2::Scalar::ONE
    } else {
      E2::Scalar::ZERO
    });

    let cc_hash = cc.squeeze(NUM_HASH_BITS);
    println!("hash in clear {:#?}", scalar_as_base::<E1>(cc_hash));

    (scalar_as_base::<E1>(cc_hash), blind)
  }
}
