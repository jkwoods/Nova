//! This module implements the Hyrax polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::NovaError,
  provider::{
    ipa_pc::{
      InnerProductArgument, InnerProductInstance, InnerProductWitness, ProverKey, VerifierKey,
    },
    pedersen::{Commitment, CommitmentKey},
    traits::DlogGroup,
  },
  spartan::polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    AbsorbInROTrait, Engine, Group, ROTrait, TranscriptReprTrait,
  },
};
use core::marker::PhantomData;
use ff::Field;
use merlin::Transcript;
use rand::rngs::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Structure that holds the blinds of a PC
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitmentBlinds<E: Engine> {
  /// Blinds
  pub blinds: Vec<E::Scalar>,
}

/// Structure that holds Poly Commits
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HyraxCommitment<E: Engine> {
  /// Commitment
  pub comm: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
}

/// Hyrax PC generators and functions to commit and prove evaluation
pub struct HyraxPolyCommit<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> TranscriptReprTrait<E::GE> for HyraxCommitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<E> AbsorbInROTrait<E> for HyraxCommitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    self.comm.absorb_in_ro(ro);
  }
}

impl<E> HyraxPolyCommit<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;

  /// Derives generators for Hyrax PC, where num_vars is the number of variables in multilinear poly
  pub fn setup(label: &'static [u8], num_vars: usize) -> (ProverKey<E>, VerifierKey<E>) {
    let (_left, right) = EqPolynomial::<E::Scalar>::compute_factored_lens(num_vars);
    let gens_v = Self::CommitmentKey {
      ck: E::GE::from_label(label, (2usize).pow(right as u32)), // TODO BLIND
    };

    let gens_s = Self::CommitmentKey {
      ck: E::GE::from_label(b"gens_s", 1), // TODO BLIND
    };

    //CommitmentGens::<G>::new_with_blinding_gen(b"gens_s", 1, &gens_v.get_blinding_gen());
    (ProverKey<E> { gens_s }, VerifierKey<E> { gens_v, gens_s }) // I think?
  }

  fn commit_inner(
    prover_key: &ProverKey<E>,
    poly: &MultilinearPolynomial<E::Scalar>,
    blinds: &[E::Scalar],
  ) -> HyraxCommitment<E> {
    let L_size = blinds.len();
    let R_size = poly.len() / L_size;

    assert_eq!(L_size * R_size, poly.len());

    let comm = (0..L_size)
      .into_par_iter()
      .map(|i| {
        E::commit(
          &self.gens_v,
          &poly.get_Z()[R_size * i..R_size * (i + 1)],
          &blinds[i],
        )
        .compress()
      })
      .collect();

    HyraxCommitment { comm }
  }

  /// Commits to a multilinear polynomial and returns commitment and blind
  pub fn commit(
    prover_key: &ProverKey<E>,
    poly: &MultilinearPolynomial<E::Scalar>,
  ) -> (HyraxCommitment<E>, HyraxCommitmentBlinds<E>) {
    let n = poly.len();
    let ell = poly.get_num_vars();
    assert_eq!(n, (2usize).pow(ell as u32));

    let (left_num_vars, right_num_vars) = EqPolynomial::<E::Scalar>::compute_factored_lens(ell);
    let L_size = (2usize).pow(left_num_vars as u32);
    let R_size = (2usize).pow(right_num_vars as u32);
    assert_eq!(L_size * R_size, n);

    let blinds = HyraxCommitmentBlinds {
      blinds: (0..L_size).map(|_| E::Scalar::random(&mut OsRng)).collect(),
    };

    (Self::commit_inner(poly, &blinds.blinds), blinds)
  }
  
  /// Proves the evaluation of polynomial at a random point r
  pub fn prove_eval(
    &self,
    poly: &MultilinearPolynomial<G::Scalar>, // defined as vector Z
    poly_com: &PolyCommit<G>,
    blinds: &PolyCommitBlinds<G>,
    r: &[G::Scalar],      // point at which the polynomial is evaluated
    Zr: &G::Scalar,       // evaluation of poly(r)
    blind_Zr: &G::Scalar, // blind for Zr
    transcript: &mut Transcript,
  ) -> Result<
    (
      InnerProductArgument<G>,
      InnerProductWitness<G>,
      CompressedCommitment<G>,
    ),
    NovaError,
  > {
    transcript.append_message(b"protocol-name", b"polynomial evaluation proof");
    poly_com.append_to_transcript(b"poly_com", transcript);

    // assert vectors are of the right size
    assert_eq!(poly.get_num_vars(), r.len());

    let (left_num_vars, right_num_vars) = EqPolynomial::<G::Scalar>::compute_factored_lens(r.len());
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
    let LZ_blind: G::Scalar = (0..L.len())
      .map(|i| blinds.blinds[i] * L[i])
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    // Translation between this stuff and IPA
    // LZ = x_vec
    // LZ_blind = r_x
    // Zr = y
    // blind_Zr = r_y
    // R = a_vec

    // Commit to LZ and Zr
    let com_LZ = G::CE::commit(&self.gens_v, &LZ, &LZ_blind);
    let com_Zr = G::CE::commit(&self.gens_s, &[*Zr], blind_Zr);

    // a dot product argument (IPA) of size R_size
    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, &com_Zr);
    let ipa_witness = InnerProductWitness::<G>::new(&LZ, &LZ_blind, Zr, blind_Zr);
    let ipa = InnerProductArgument::<G>::prove(
      &self.gens_v,
      &self.gens_s,
      &ipa_instance,
      &ipa_witness,
      transcript,
    )?;

    Ok((ipa, ipa_witness, com_Zr.compress()))
  }

  /// Verifies the proof showing the evaluation of a committed polynomial at a random point
  pub fn verify_eval(
    &self,
    r: &[G::Scalar], // point at which the polynomial was evaluated
    poly_com: &PolyCommit<G>,
    com_Zr: &CompressedCommitment<G>, // commitment to evaluation
    ipa: &InnerProductArgument<G>,
    transcript: &mut Transcript,
  ) -> Result<(), NovaError> {
    transcript.append_message(b"protocol-name", b"polynomial evaluation proof");
    poly_com.append_to_transcript(b"poly_com", transcript);

    // compute L and R
    let eq = EqPolynomial::new(r.to_vec());
    let (L, R) = eq.compute_factored_evals();

    // compute a weighted sum of commitments and L
    let C_preprocessed: Vec<G::PreprocessedGroupElement> = poly_com
      .comm
      .iter()
      .map(|pt| pt.decompress().unwrap().reinterpret_as_generator())
      .collect();

    let gens = CommitmentGens::<G>::from_preprocessed(C_preprocessed);
    let com_LZ = G::CE::commit(&gens, &L, &G::Scalar::zero()); // computes MSM of commitment and L

    let com_Zr_decomp = com_Zr.decompress()?;

    let ipa_instance = InnerProductInstance::<G>::new(&com_LZ, &R, &com_Zr_decomp);

    ipa.verify(
      &self.gens_v,
      &self.gens_s,
      R.len(),
      &ipa_instance,
      transcript,
    )
  }*/
}

#[cfg(test)]
mod tests {
  use super::*;
  type G = pasta_curves::pallas::Point;
  use rand::rngs::OsRng;

  fn inner_product<T>(a: &[T], b: &[T]) -> T
  where
    T: Field + Send + Sync,
  {
    assert_eq!(a.len(), b.len());
    (0..a.len())
      .into_par_iter()
      .map(|i| a[i] * b[i])
      .reduce(T::zero, |x, y| x + y)
  }

  fn evaluate_with_LR(
    Z: &[<G as Group>::Scalar],
    r: &[<G as Group>::Scalar],
  ) -> <G as Group>::Scalar {
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
          .fold(<G as Group>::Scalar::zero(), |acc, item| acc + item)
      })
      .collect::<Vec<<G as Group>::Scalar>>();

    // compute dot product between LZ and R
    inner_product(&LZ, &R)
  }

  fn to_scalar(x: usize) -> <G as Group>::Scalar {
    (0..x)
      .map(|_i| <G as Group>::Scalar::one())
      .fold(<G as Group>::Scalar::zero(), |acc, item| acc + item)
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

    let prover_gens = HyraxPC::new(num_vars, b"poly_test");
    let (poly_comm, blinds) = prover_gens.commit(&poly);

    let mut prover_transcript = Transcript::new(b"example");

    let blind_eval = <G as Group>::Scalar::random(&mut OsRng);

    let (ipa_proof, _ipa_witness, comm_eval): (InnerProductArgument<G>, InnerProductWitness<G>, _) =
      prover_gens
        .prove_eval(
          &poly,
          &poly_comm,
          &blinds,
          &r,
          &eval,
          &blind_eval,
          &mut prover_transcript,
        )
        .unwrap();

    // Verifier actions

    let verifier_gens = HyraxPC::new(num_vars, b"poly_test");
    let mut verifier_transcript = Transcript::new(b"example");

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
