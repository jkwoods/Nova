//! This module provides an incremental commitment scheme
use crate::{
  constants::NUM_HASH_BITS,
  gadgets::utils::scalar_as_base,
  provider::{
    pedersen::CommitmentKeyExtTrait, poseidon::PoseidonConstantsCircuit, traits::DlogGroup,
    PoseidonRO,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    ROCircuitTrait, ROConstants, ROTrait,
  },
  Commitment, CommitmentKey, Engine,
};
use ff::Field;
use rand_core::OsRng;
use std::{io::BufReader, marker::PhantomData, path::Path};

/// Incremental Commitment Scheme Generators
pub struct Incremental<E1: Engine, E2: Engine>
where
  E1::GE: DlogGroup,
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  /// kzg gens  
  pub kzg_gen: CommitmentKey<E1>,
  pos_constants: ROConstants<E1>,
  _p: PhantomData<E2>,
}

impl<E1: Engine, E2: Engine> Incremental<E1, E2>
where
  E1::GE: DlogGroup,
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
{
  /// setup
  pub fn setup<P: AsRef<Path>>(key_len: usize, path: P) -> Self {
    // from ppot file
    let mut reader = BufReader::new(std::fs::File::open(path).unwrap());

    let kzg_gen: CommitmentKey<E1> = E1::CE::load_setup(&mut reader, b"ppot", key_len).unwrap();

    //let kzg_gen = E1::CE::setup(b"ck", key_len);
    let pos_constants: ROConstants<E1> = ROConstants::<E1>::default();

    Incremental::<E1, E2> {
      kzg_gen,
      pos_constants,
      _p: PhantomData::default(),
    }
  }

  /// inner kzg commit
  pub fn inner_commit(&self, w: &[E1::Scalar]) -> (Commitment<E1>, E1::Scalar) {
    let blind = E1::Scalar::random(&mut OsRng);

    let kzg_cmt = E1::CE::commit(&self.kzg_gen, w, &blind);

    (kzg_cmt, blind)
  }

  /// incremental hash
  pub fn hash(&self, c_i: Option<E2::Scalar>, kzg_cmt: &Commitment<E1>) -> E2::Scalar {
    let mut cc = E1::RO::new(self.pos_constants.clone());

    if c_i.is_none() {
      cc.absorb(E2::Scalar::ZERO);
    } else {
      cc.absorb(c_i.unwrap());
    }

    let kzg_coords = kzg_cmt.to_coordinates();
    //println!("x {:#?}", kzg_coords.0);

    cc.absorb(kzg_coords.0);
    cc.absorb(kzg_coords.1);
    cc.absorb(if kzg_coords.2 {
      E2::Scalar::ONE
    } else {
      E2::Scalar::ZERO
    });

    let cc_hash = cc.squeeze(NUM_HASH_BITS);
    //println!("hash in clear {:#?}", scalar_as_base::<E1>(cc_hash));

    cc_hash
  }

  /// commit incrementally to chunk of list
  pub fn commit(&self, c_i: Option<E2::Scalar>, w: &[E1::Scalar]) -> (E2::Scalar, E1::Scalar) {
    let mut cc = E1::RO::new(self.pos_constants.clone());

    if c_i.is_none() {
      cc.absorb(E2::Scalar::ZERO);
    } else {
      cc.absorb(c_i.unwrap());
    }

    let blind = E1::Scalar::random(&mut OsRng);
    //println!("committing to {:#?} with blind {:#?}", w, blind);
    let kzg_cmt = E1::CE::commit(&self.kzg_gen, w, &blind);

    //println!("cmt in clear {:#?}", kzg_cmt);

    let kzg_coords = kzg_cmt.to_coordinates();
    //println!("x {:#?}", kzg_coords.0);

    cc.absorb(kzg_coords.0);
    cc.absorb(kzg_coords.1);
    cc.absorb(if kzg_coords.2 {
      E2::Scalar::ONE
    } else {
      E2::Scalar::ZERO
    });

    let cc_hash = cc.squeeze(NUM_HASH_BITS);
    //println!("hash in clear {:#?}", scalar_as_base::<E1>(cc_hash));

    (cc_hash, blind)
    // (scalar_as_base::<E1>(cc_hash), blind)
  }
}
