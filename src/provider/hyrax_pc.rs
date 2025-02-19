//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  provider::{
    pedersen::CommitmentKeyExtTrait,
    traits::DlogGroup,
    zk_ipa_pc::{InnerProductArgument, InnerProductInstance, InnerProductWitness},
  },
  spartan::polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  traits::{
    commitment::{AffineTrait, CommitmentEngineTrait, GetGeneratorsTrait},
    Engine, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, NovaError, CE,
};
use ff::Field;
use rand_core::OsRng;
use rayon::prelude::*;
//use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};

/// Structure that holds the blinds of a PC
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PolyCommitBlinds<E: Engine> {
  /// Blinds
  pub blinds: Vec<E::Scalar>,
}

/// Structure that holds Poly Commits
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PolyCommit<E: Engine>
where
  E::GE: DlogGroup,
{
  /// Commitment
  pub comm: Vec<Commitment<E>>,
}

/// Hyrax PC generators and functions to commit and prove evaluation
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxPC<E: Engine>
where
  E::GE: DlogGroup,
{
  /// generator for vectors
  pub ck_v: CommitmentKey<E>, // generator for vectors
  /// generator for scalars (eval)
  pub ck_s: CommitmentKey<E>, // generator for scalars (eval)
}

impl<E: Engine> TranscriptReprTrait<E::GE> for PolyCommit<E>
where
  E::GE: DlogGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut bytes = Vec::new();
    for c in &self.comm {
      bytes.append(&mut c.to_transcript_bytes());
    }
    bytes
  }
}

impl<E: Engine> HyraxPC<E>
where
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E> + GetGeneratorsTrait<E>,
  Commitment<E>: AffineTrait<E>,
{
  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  pub fn setup(label: &'static [u8], num_vars: usize) -> Self {
    let right = num_vars - num_vars / 2;

    let gens = E::GE::from_label(label, (2usize).pow(right as u32) + 2);
    let (h, gens) = gens.split_first().unwrap();
    let (g_s, g_v) = gens.split_first().unwrap();

    let ck_v = CommitmentKey::<E>::from_gens(g_v.to_vec(), Some(h.clone()));
    let ck_s = CommitmentKey::<E>::from_gens(vec![g_s.clone()], Some(h.clone()));

    HyraxPC { ck_v, ck_s }
  }

  /// Same as setup, with a preset eval generator
  pub fn setup_with_ck_s(label: &'static [u8], num_vars: usize, ck_s: CommitmentKey<E>) -> Self {
    let right = num_vars - num_vars / 2;
    let gens = E::GE::from_label(label, (2usize).pow(right as u32));
    let ck_v = CommitmentKey::<E>::from_gens(gens, Some(ck_s.get_h().clone()));

    HyraxPC { ck_v, ck_s }
  }

  fn commit_inner(
    &self,
    poly: &MultilinearPolynomial<E::Scalar>,
    blinds: &[E::Scalar],
  ) -> PolyCommit<E> {
    let L_size = blinds.len();
    let R_size = poly.len() / L_size;

    assert_eq!(L_size * R_size, poly.len());

    let comm = (0..L_size)
      .into_par_iter()
      .map(|i| {
        CE::<E>::commit(
          &self.ck_v,
          &poly.Z[R_size * i..R_size * (i + 1)],
          &blinds[i],
        )
      })
      .collect();

    PolyCommit { comm }
  }

  /// Commits to a multilinear polynomial and returns commitment and blind
  pub fn commit(
    &self,
    poly: &MultilinearPolynomial<E::Scalar>,
  ) -> (PolyCommit<E>, PolyCommitBlinds<E>) {
    let n = poly.len();
    let ell = poly.get_num_vars();
    assert_eq!(n, (2usize).pow(ell as u32));

    let (left_num_vars, right_num_vars) = (ell / 2, ell - ell / 2);

    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);
    assert_eq!(L_size * R_size, n);

    let blinds = PolyCommitBlinds {
      blinds: (0..L_size).map(|_| E::Scalar::random(&mut OsRng)).collect(),
    };

    (self.commit_inner(poly, &blinds.blinds), blinds)
  }

  const fn protocol_name() -> &'static [u8] {
    b"polynomial evaluation proof"
  }

  /// Proves the evaluation of polynomial at a random point r
  pub fn prove_eval(
    &self,
    poly: &MultilinearPolynomial<E::Scalar>, // defined as vector Z
    poly_com: &PolyCommit<E>,
    blinds: &PolyCommitBlinds<E>,
    r: &[E::Scalar], // point at which the polynomial is evaluated
    Zr: &E::Scalar,  // evaluation of poly(r)
    com_Zr: &Commitment<E>,
    blind_Zr: &E::Scalar, // blind for Zr
    transcript: &mut E::TE,
  ) -> Result<
    (
      InnerProductArgument<E>,
      InnerProductWitness<E>,
      //Commitment<E>,
    ),
    NovaError,
  > {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"poly_com", poly_com);

    // assert vectors are of the right size
    assert_eq!(poly.get_num_vars(), r.len());

    let (left_num_vars, right_num_vars) = (r.len() / 2, r.len() - r.len() / 2);
    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);

    assert_eq!(blinds.blinds.len(), L_size);

    // compute the L and R vectors (these depend only on the public challenge r so they are public)
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();
    assert_eq!(L.len(), L_size);
    assert_eq!(R.len(), R_size);

    // compute the vector underneath L*Z and the L*blinds
    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = poly.bound(&L);
    let LZ_blind: E::Scalar = (0..L.len())
      .map(|i| blinds.blinds[i] * L[i])
      .fold(E::Scalar::ZERO, |acc, item| acc + item);

    // Translation between this stuff and IPA
    // LZ = x_vec
    // LZ_blind = r_x
    // Zr = y
    // blind_Zr = r_y
    // R = a_vec

    // Commit to LZ and Zr
    let com_LZ = CE::<E>::commit(&self.ck_v, &LZ, &LZ_blind);
    //let com_Zr = CE::<E>::commit(&self.ck_s, &[*Zr], blind_Zr);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<E>::new(&com_LZ, &R, com_Zr);
    let ipa_witness = InnerProductWitness::<E>::new(&LZ, &LZ_blind, Zr, blind_Zr);
    let ipa = InnerProductArgument::<E>::prove(
      &self.ck_v,
      &self.ck_s,
      &ipa_instance,
      &ipa_witness,
      transcript,
    )?;

    Ok((ipa, ipa_witness))
  }

  /// Verifies the proof showing the evaluation of a committed polynomial at a random point
  pub fn verify_eval(
    &self,
    r: &[E::Scalar], // point at which the polynomial was evaluated
    poly_com: &PolyCommit<E>,
    com_Zr: &Commitment<E>, // commitment to evaluation
    ipa: &InnerProductArgument<E>,
    transcript: &mut E::TE,
  ) -> Result<(), NovaError> {
    transcript.dom_sep(Self::protocol_name());
    transcript.absorb(b"poly_com", poly_com);

    // compute L and R
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    // compute a weighted sum of commitments and L
    let C_preprocessed: Vec<<E::GE as DlogGroup>::AffineGroupElement> =
      poly_com.comm.iter().map(|pt| pt.affine()).collect();

    let gens = CommitmentKey::<E>::from_gens(C_preprocessed, None);
    let com_LZ = CE::<E>::commit(&gens, &L, &E::Scalar::ZERO); // computes MSM of commitment and L

    let ipa_instance = InnerProductInstance::<E>::new(&com_LZ, &R, &com_Zr);

    ipa.verify(&self.ck_v, &self.ck_s, R.len(), &ipa_instance, transcript)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  type E = crate::provider::PallasEngine;
  use rand::rngs::OsRng;

  fn evaluate_with_LR(
    Z: &[<E as Engine>::Scalar],
    r: &[<E as Engine>::Scalar],
  ) -> <E as Engine>::Scalar {
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    let ell = r.len();
    // ensure ell is even
    assert!(ell % 2 == 0);

    // compute n = 2^\ell
    let n = (2usize).pow(ell as u32);

    // compute m = sqrt(n) = 2^{\ell/2}
    let m = (n as f64).sqrt() as usize;

    // compute vector-matrix product between L and Z viewed as a matrix
    let LZ = (0..m)
      .map(|i| {
        (0..m)
          .map(|j| L[j] * Z[j * m + i])
          .fold(<E as Engine>::Scalar::ZERO, |acc, item| acc + item)
      })
      .collect::<Vec<<E as Engine>::Scalar>>();

    // compute dot product between LZ and R
    InnerProductArgument::<E>::inner_product(&LZ, &R)
  }

  fn to_scalar(x: usize) -> <E as Engine>::Scalar {
    (0..x)
      .map(|_i| <E as Engine>::Scalar::ONE)
      .fold(<E as Engine>::Scalar::ZERO, |acc, item| acc + item)
  }

  #[test]
  fn check_polynomial_evaluation() {
    // Z = [1, 2, 1, 4]
    let Z = vec![to_scalar(1), to_scalar(2), to_scalar(1), to_scalar(4)];

    // r = [4,3]
    let r = vec![to_scalar(4), to_scalar(3)];

    let eval_with_LR = evaluate_with_LR(&Z, &r);
    let poly = MultilinearPolynomial::new(Z);

    let eval = poly.evaluate(&r);
    assert_eq!(eval, to_scalar(28));
    assert_eq!(eval_with_LR, eval);
  }

  #[test]
  fn check_hyrax_pc_commit() {
    let Z = vec![to_scalar(1), to_scalar(2), to_scalar(1), to_scalar(4)];

    let poly = MultilinearPolynomial::new(Z);

    // Public stuff
    let num_vars = 2;
    assert_eq!(num_vars, poly.get_num_vars());
    let r = vec![to_scalar(4), to_scalar(3)]; // r = [4,3]

    // Prover actions
    let eval = poly.evaluate(&r);
    assert_eq!(eval, to_scalar(28));

    let prover_gens = HyraxPC::setup(b"poly_test", num_vars);
    let (poly_comm, blinds) = prover_gens.commit(&poly);

    let mut prover_transcript = <E as Engine>::TE::new(b"example");

    let blind_eval = <E as Engine>::Scalar::random(&mut OsRng);
    let comm_eval = CE::<E>::commit(&prover_gens.ck_s, &[eval], &blind_eval);

    let (ipa_proof, _ipa_witness): (InnerProductArgument<E>, InnerProductWitness<E>) = prover_gens
      .prove_eval(
        &poly,
        &poly_comm,
        &blinds,
        &r,
        &eval,
        &comm_eval,
        &blind_eval,
        &mut prover_transcript,
      )
      .unwrap();

    // Verifier actions

    let verifier_gens = HyraxPC::setup(b"poly_test", num_vars);
    let mut verifier_transcript = <E as Engine>::TE::new(b"example");

    let res = verifier_gens.verify_eval(
      &r,
      &poly_comm,
      &comm_eval,
      &ipa_proof,
      &mut verifier_transcript,
    );
    assert!(res.is_ok());
  }
}
