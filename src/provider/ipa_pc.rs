//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  spartan::polys::eq::EqPolynomial,
  traits::{
    commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
    TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, CE,
};
use core::iter;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Provides an implementation of the prover key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine> {
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of the verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine> {
  ck_v: CommitmentKey<E>,
  ck_s: CommitmentKey<E>,
}

/*/// A type that holds unsplitting proof information
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct UnsplitProof<E: Engine> {
  comm_W: <E::CE as CommitmentEngineTrait<E>>::Commitment,
  big_ipa: InnerProductArgument<E>,
  big_eval: E::Scalar,
  small: Vec<(E::Scalar, InnerProductArgument<E>)>,
}

impl<E: Engine> UnsplitProofTrait<E> for UnsplitProof<E> {
  fn get_comm_W(&self) -> <E::CE as CommitmentEngineTrait<E>>::Commitment {
    return self.comm_W;
  }
}
*/

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;
  type EvaluationArgument = InnerProductArgument<E>;

  fn setup(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let ck_c = E::CE::setup(b"ipa", 1);

    let pk = ProverKey { ck_s: ck_c.clone() };
    let vk = VerifierKey {
      ck_v: ck.clone(),
      ck_s: ck_c,
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    poly: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    InnerProductArgument::prove(ck, &pk.ck_s, &u, &w, transcript)
  }

  /*  fn prove_unsplit_witnesses(
      ck: &CommitmentKey<E>,
      pk: &Self::ProverKey,
      U: &RelaxedR1CSInstance<E>,
      W: &RelaxedR1CSWitness<E>,
    ) -> Result<
      (
        RelaxedR1CSInstance<E>,
        RelaxedR1CSWitness<E>,
        Self::UnsplitProof,
      ),
      NovaError,
    > {
      let wits: Vec<E::Scalar> = W.W.iter().flatten().cloned().collect();
      println!("WITS BIG LEN {:#?}", wits.len());
      let comm_W = CE::<E>::commit(ck, &wits, &E::Scalar::ZERO);

      let mut transcript = <E as Engine>::TE::new(b"unsplit");
      transcript.absorb(b"W", &comm_W);

      // get rs from transcript
      let mut r = Vec::new();
      for _i in 0..W.W.len() {
        r.push(transcript.squeeze(b"r")?);
      }

      println!("P rs {:#?}", r.clone());

      // prove IPAs
      let mut small = Vec::new();
      let mut big_r = Vec::new();
      let mut eval_sum = E::Scalar::ZERO;
      let mut ck_big = ck.clone();
      let mut ck_small;

      for i in 0..W.W.len() {
        let mut padded_wit = W.W[i].clone();
        padded_wit.extend(vec![
          E::Scalar::ZERO;
          W.W[i].len().next_power_of_two() - W.W[i].len()
        ]);
        let small_r = vec![r[i]; padded_wit.len()];

        let small_eval = inner_product(&padded_wit, &small_r);
        let u = InnerProductInstance::new(&U.comm_W[i], &small_r, &small_eval);
        let w = InnerProductWitness::new(&padded_wit);
        (ck_small, ck_big) = E::CE::split_key_at(&ck_big, W.W[i].len());

        /*    println!(
                "P small r {:#?}, comm_W_i {:#?} eval {:#?} ck {:#?} ck_s {:#?}, commited small_r {:#?}",
                small_r.clone(),
                U.comm_W[i].clone(),
                small_eval.clone(),
                ck_small.clone(),
                pk.ck_s.clone(),
                CE::<E>::commit(&ck_small, &padded_wit, &E::Scalar::ZERO),
              );

        use crate::traits::commitment::Len;
        println!(
          "PROVING small r: {:#?}, w: {:#?}, ck_small: {:#?}",
          small_r.len(),
          padded_wit.len(),
          ck_small.length()
        );*/

        let small_ipa = InnerProductArgument::prove(&ck_small, &pk.ck_s, &u, &w, &mut transcript)?;

        eval_sum += small_eval;

        small.push((small_eval, small_ipa));
        big_r.extend(vec![r[i]; W.W[i].len()]);
      }

      let big_eval = inner_product(&wits, &big_r);
      assert_eq!(big_eval, eval_sum);
      let u = InnerProductInstance::new(&comm_W, &big_r, &big_eval);
      let w = InnerProductWitness::new(&wits);
      let big_ipa = InnerProductArgument::prove(ck, &pk.ck_s, &u, &w, &mut transcript)?;

      Ok((
        RelaxedR1CSInstance {
          comm_W: vec![comm_W],
          comm_E: U.comm_E.clone(),
          X: U.X.clone(),
          u: U.u,
        },
        RelaxedR1CSWitness {
          W: vec![wits],
          r_W: vec![W.r_W.iter().sum()],
          E: W.E.clone(),
          r_E: W.r_E.clone(),
        },
        UnsplitProof {
          comm_W,
          big_ipa,
          big_eval,
          small,
        },
      ))
    }
  */
  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.verify(
      &vk.ck_v,
      &vk.ck_s,
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    Ok(())
  }

  /*fn verify_unsplit_witnesses(
    vk: &Self::VerifierKey,
    p: &Self::UnsplitProof,
    U: &RelaxedR1CSInstance<E>,
    S: &R1CSShape<E>,
  ) -> Result<RelaxedR1CSInstance<E>, NovaError> {
    assert_eq!(U.comm_W.len(), p.small.len());

    let mut transcript = <E as Engine>::TE::new(b"unsplit");
    transcript.absorb(b"W", &p.comm_W);

    // get rs from transcript
    let mut r = Vec::new();
    for _i in 0..U.comm_W.len() {
      r.push(transcript.squeeze(b"r")?);
    }
    println!("V rs {:#?}", r.clone());

    // verify IPAs
    let mut big_r = Vec::new();
    let mut eval_sum = E::Scalar::ZERO;
    let mut ck_big = vk.ck_v.clone();
    let mut ck_small;

    for (i, (eval, ipa)) in p.small.iter().enumerate() {
      let small_r = vec![r[i]; S.num_split_vars[i].next_power_of_two()];
      let u = InnerProductInstance::new(&U.comm_W[i], &small_r, eval);
      (ck_small, ck_big) = E::CE::split_key_at(&ck_big, S.num_split_vars[i]);

      /*  use crate::traits::commitment::Len;
            println!(
              "V small r {:#?}, comm_W_i {:#?} eval {:#?}, ck {:#?}, ck_s {:#?}",
              small_r.clone(),
              U.comm_W[i].clone(),
              eval.clone(),
              ck_small.clone(),
              vk.ck_s.clone(),
            );
      */
      ipa.verify(&ck_small, &vk.ck_s, small_r.len(), &u, &mut transcript)?;

      eval_sum += eval;
      big_r.extend(vec![r[i]; S.num_split_vars[i]]);
    }

    println!("small ipas verified");

    let u = InnerProductInstance::new(&p.comm_W, &big_r, &p.big_eval);
    p.big_ipa
      .verify(&vk.ck_v, &vk.ck_s, big_r.len(), &u, &mut transcript)?;

    println!("big ipa verified");

    if eval_sum != p.big_eval {
      return Err(NovaError::UnsplitError);
    }

    Ok(RelaxedR1CSInstance {
      comm_W: vec![p.comm_W],
      comm_E: U.comm_E.clone(),
      X: U.X.clone(),
      u: U.u,
    })
  }*/
}

pub(crate) fn inner_product<T: Field + Send + Sync>(a: &[T], b: &[T]) -> T {
  assert_eq!(a.len(), b.len());
  (0..a.len())
    .into_par_iter()
    .map(|i| a[i] * b[i])
    .reduce(|| T::ZERO, |x, y| x + y)
}

/// An inner product instance consists of a commitment to a vector `a` and another vector `b`
/// and the claim that c = <a, b>.
pub(crate) struct InnerProductInstance<E: Engine> {
  comm_a_vec: Commitment<E>,
  b_vec: Vec<E::Scalar>,
  c: E::Scalar,
}

impl<E> InnerProductInstance<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  pub fn new(comm_a_vec: &Commitment<E>, b_vec: &[E::Scalar], c: &E::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: *comm_a_vec,
      b_vec: b_vec.to_vec(),
      c: *c,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for InnerProductInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.b_vec as in our context it is produced from the transcript
    [
      self.comm_a_vec.to_transcript_bytes(),
      self.c.to_transcript_bytes(),
    ]
    .concat()
  }
}

pub(crate) struct InnerProductWitness<E: Engine> {
  a_vec: Vec<E::Scalar>,
}

impl<E: Engine> InnerProductWitness<E> {
  pub fn new(a_vec: &[E::Scalar]) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine> {
  L_vec: Vec<Commitment<E>>,
  R_vec: Vec<Commitment<E>>,
  a_hat: E::Scalar,
}

impl<E> InnerProductArgument<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  fn prove(
    ck: &CommitmentKey<E>,
    ck_c: &CommitmentKey<E>,
    U: &InnerProductInstance<E>,
    W: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, NovaError> {
    transcript.dom_sep(Self::protocol_name());

    let (ck, _) = ck.split_at(U.b_vec.len());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[E::Scalar],
                       b_vec: &[E::Scalar],
                       ck: &CommitmentKey<E>,
                       transcript: &mut E::TE|
     -> Result<
      (
        Commitment<E>,
        Commitment<E>,
        Vec<E::Scalar>,
        Vec<E::Scalar>,
        CommitmentKey<E>,
      ),
      NovaError,
    > {
      let n = a_vec.len();
      let (ck_L, ck_R) = ck.split_at(n / 2);

      let c_L = inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]);
      let c_R = inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]);

      let L = CE::<E>::commit(
        &ck_R.combine(&ck_c),
        &a_vec[0..n / 2]
          .iter()
          .chain(iter::once(&c_L))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &E::Scalar::ZERO,
      );
      let R = CE::<E>::commit(
        &ck_L.combine(&ck_c),
        &a_vec[n / 2..n]
          .iter()
          .chain(iter::once(&c_R))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &E::Scalar::ZERO,
      );

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);

      let r = transcript.squeeze(b"r")?;
      let r_inverse = r.invert().unwrap();

      // fold the left half and the right half
      let a_vec_folded = a_vec[0..n / 2]
        .par_iter()
        .zip(a_vec[n / 2..n].par_iter())
        .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
        .collect::<Vec<E::Scalar>>();

      let b_vec_folded = b_vec[0..n / 2]
        .par_iter()
        .zip(b_vec[n / 2..n].par_iter())
        .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
        .collect::<Vec<E::Scalar>>();

      let ck_folded = ck.fold(&r_inverse, &r);

      Ok((L, R, a_vec_folded, b_vec_folded, ck_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<Commitment<E>> = Vec::new();
    let mut R_vec: Vec<Commitment<E>> = Vec::new();

    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut ck = ck;
    for _i in 0..usize::try_from(U.b_vec.len().ilog2()).unwrap() {
      let (L, R, a_vec_folded, b_vec_folded, ck_folded) =
        prove_inner(&a_vec, &b_vec, &ck, transcript)?;
      L_vec.push(L);
      R_vec.push(R);

      a_vec = a_vec_folded;
      b_vec = b_vec_folded;
      ck = ck_folded;
    }

    Ok(InnerProductArgument {
      L_vec,
      R_vec,
      a_hat: a_vec[0],
    })
  }

  fn verify(
    &self,
    ck: &CommitmentKey<E>,
    ck_c: &CommitmentKey<E>,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), NovaError> {
    let (ck, _) = ck.split_at(U.b_vec.len());

    transcript.dom_sep(Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    let P = U.comm_a_vec + CE::<E>::commit(&ck_c, &[U.c], &E::Scalar::ZERO);

    let batch_invert = |v: &[E::Scalar]| -> Result<Vec<E::Scalar>, NovaError> {
      let mut products = vec![E::Scalar::ZERO; v.len()];
      let mut acc = E::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // return error if acc is zero
      acc = match Option::from(acc.invert()) {
        Some(inv) => inv,
        None => return Err(NovaError::InternalError),
      };

      // compute the inverse once for all entries
      let mut inv = vec![E::Scalar::ZERO; v.len()];
      for i in (0..v.len()).rev() {
        let tmp = acc * v[i];
        inv[i] = products[i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute a vector of public coins using self.L_vec and self.R_vec
    let r = (0..self.L_vec.len())
      .map(|i| {
        transcript.absorb(b"L", &self.L_vec[i]);
        transcript.absorb(b"R", &self.R_vec[i]);
        transcript.squeeze(b"r")
      })
      .collect::<Result<Vec<E::Scalar>, NovaError>>()?;

    // precompute scalars necessary for verification
    let r_square: Vec<E::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r[i] * r[i])
      .collect();
    let r_inverse = batch_invert(&r)?;
    let r_inverse_square: Vec<E::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r_inverse[i] * r_inverse[i])
      .collect();

    // compute the vector with the tensor structure
    let s = {
      let mut s = vec![E::Scalar::ZERO; n];
      s[0] = {
        let mut v = E::Scalar::ONE;
        for r_inverse_i in r_inverse {
          v *= r_inverse_i;
        }
        v
      };
      for i in 1..n {
        let pos_in_r = (31 - (i as u32).leading_zeros()) as usize;
        s[i] = s[i - (1 << pos_in_r)] * r_square[(self.L_vec.len() - 1) - pos_in_r];
      }
      s
    };

    let ck_hat = {
      let c = CE::<E>::commit(&ck, &s, &E::Scalar::ZERO);
      CommitmentKey::<E>::reinterpret_commitments_as_ck(&[c])?
    };

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = {
        let ck_L = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.L_vec)?;
        let ck_R = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.R_vec)?;
        let ck_P = CommitmentKey::<E>::reinterpret_commitments_as_ck(&[P])?;
        ck_L.combine(&ck_R).combine(&ck_P)
      };

      CE::<E>::commit(
        &ck_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&E::Scalar::ONE))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &E::Scalar::ZERO,
      )
    };

    if P_hat
      == CE::<E>::commit(
        &ck_hat.combine(&ck_c),
        &[self.a_hat, self.a_hat * b_hat],
        &E::Scalar::ZERO,
      )
    {
      Ok(())
    } else {
      Err(NovaError::InvalidPCS)
    }
  }
}
