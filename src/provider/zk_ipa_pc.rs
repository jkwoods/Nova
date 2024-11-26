//! ZK IPA for Hyrax
use crate::{
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  traits::{
    commitment::{AffineTrait, CommitmentEngineTrait, GetGeneratorsTrait},
    Engine, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, CE,
};
use core::iter;
use ff::Field;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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

/// An inner product instance consists of a commitment to a vector `x`,
/// a commitment to a scalar `y`, another vector `a`
/// and the claim that y = <x, a>.
pub struct InnerProductInstance<E: Engine> {
  comm_x_vec: Commitment<E>,
  a_vec: Vec<E::Scalar>,
  comm_y: Commitment<E>,
}

impl<E> InnerProductInstance<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  /// new zk ipa
  pub fn new(comm_x_vec: &Commitment<E>, a_vec: &[E::Scalar], comm_y: &Commitment<E>) -> Self {
    InnerProductInstance {
      comm_x_vec: *comm_x_vec,
      a_vec: a_vec.to_vec(),
      comm_y: *comm_y,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for InnerProductInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.b_vec as in our context it is produced from the transcript
    // - CHECK TODO if this is still true for us
    [
      self.comm_x_vec.to_transcript_bytes(),
      //      self.a_vec.to_transcript_bytes(),
      self.comm_y.to_transcript_bytes(),
    ]
    .concat()
  }
}

/// An innter produce witness consists of the vector `x`, evaluation `y`, and blinds for `x` and `y`
pub struct InnerProductWitness<E: Engine> {
  x_vec: Vec<E::Scalar>,
  r_x: E::Scalar,
  y: E::Scalar,
  r_y: E::Scalar,
}

impl<E: Engine> InnerProductWitness<E> {
  /// new IPA witness
  pub(crate) fn new(x_vec: &[E::Scalar], r_x: &E::Scalar, y: &E::Scalar, r_y: &E::Scalar) -> Self {
    InnerProductWitness {
      x_vec: x_vec.to_vec(),
      r_x: *r_x,
      y: *y,
      r_y: *r_y,
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine> {
  P_L_vec: Vec<Commitment<E>>,
  P_R_vec: Vec<Commitment<E>>,
  delta: Commitment<E>,
  beta: Commitment<E>,
  z_1: E::Scalar,
  z_2: E::Scalar,
}

impl<E> InnerProductArgument<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E> + GetGeneratorsTrait<E>,
  Commitment<E>: AffineTrait<E>,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  /// proves IPA
  pub fn prove(
    ck: &CommitmentKey<E>,
    ck_y: &CommitmentKey<E>,
    U: &InnerProductInstance<E>,
    W: &InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, NovaError> {
    transcript.dom_sep(Self::protocol_name());

    let (ck, _) = ck.split_at(U.a_vec.len());

    if U.a_vec.len() != W.x_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let chal = transcript.squeeze(b"r")?;
    let ck_y = ck_y.scale(&chal);

    // two vectors to hold the logarithmic number of group elements, and their masks
    let mut P_L_vec: Vec<Commitment<E>> = Vec::new();
    let mut P_R_vec: Vec<Commitment<E>> = Vec::new();
    // Step 1 in Hyrax's Figure 7. The prover doesn't need P explicitly, so we don't
    // need to compute it. We just compute the randomness used in the commitment.
    let mut r_P = W.r_x + W.r_y * chal;

    // we create mutable copies of vectors and generators
    let mut x_vec = W.x_vec.to_vec();
    let mut a_vec = U.a_vec.to_vec();
    let mut ck = ck.clone();
    let mut y = W.y;
    for _i in 0..(U.a_vec.len() as f64).log2() as usize {
      let (r_P_prime, P_L, P_R, y_prime, x_vec_prime, a_vec_prime, ck_prime) =
        Self::bullet_reduce_prover(&r_P, &x_vec, &a_vec, &y, &ck, &ck_y, transcript)?;
      P_L_vec.push(P_L);
      P_R_vec.push(P_R);

      r_P = r_P_prime;
      y = y_prime;
      x_vec = x_vec_prime;
      a_vec = a_vec_prime;
      ck = ck_prime;
    }

    assert_eq!(a_vec.len(), 1);
    // This is after the recursive calls to bullet_reduce in Hyrax
    let r_P_hat = r_P;
    let y_hat = y;
    let a_hat = a_vec[0];
    let g_hat = ck;
    let d = E::Scalar::random(&mut OsRng);
    let r_delta = E::Scalar::random(&mut OsRng);
    let r_beta = E::Scalar::random(&mut OsRng);
    let delta = CE::<E>::commit(&g_hat, &[d], &r_delta);
    let beta = CE::<E>::commit(&ck_y, &[d], &r_beta);

    transcript.absorb(b"beta", &beta);
    transcript.absorb(b"delta", &delta);

    let chal = transcript.squeeze(b"chal_z")?;

    let z_1 = d + chal * y_hat;
    let z_2 = a_hat * ((chal * r_P_hat) + r_beta) + r_delta;
    Ok(InnerProductArgument {
      P_L_vec,
      P_R_vec,
      delta,
      beta,
      z_1,
      z_2,
    })
  }

  /// verfies IPA
  pub fn verify(
    &self,
    ck: &CommitmentKey<E>,
    ck_y: &CommitmentKey<E>,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), NovaError> {
    let (ck, _) = ck.split_at(U.a_vec.len()); // need?

    transcript.dom_sep(Self::protocol_name());
    if U.a_vec.len() != n
      || n != (1 << self.P_L_vec.len())
      || self.P_L_vec.len() != self.P_R_vec.len()
      || self.P_L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let chal = transcript.squeeze(b"r")?;
    let ck_y = ck_y.scale(&chal);

    let P = U.comm_x_vec + U.comm_y * chal;

    let a_vec = U.a_vec.clone();

    // calculate all the exponent challenges (s) and inverses at once
    let (mut u_sq, mut u_inv_sq, s) = self.verification_scalars(n, transcript)?;

    // do all the exponentiations at once (Hyrax, Fig. 7, step 4, all rounds)
    let g_hat = E::GE::vartime_multiscalar_mul(&s, ck.get_ck());
    let a = Self::inner_product(&a_vec[..], &s[..]);

    let mut Ls: Vec<<E::GE as DlogGroup>::AffineGroupElement> =
      self.P_L_vec.iter().map(|p| p.affine()).collect();
    let mut Rs: Vec<<E::GE as DlogGroup>::AffineGroupElement> =
      self.P_R_vec.iter().map(|p| p.affine()).collect();
    Ls.append(&mut Rs);
    Ls.push(P.affine());
    u_sq.append(&mut u_inv_sq);
    u_sq.push(E::Scalar::ONE);

    let P_comm = E::GE::vartime_multiscalar_mul(&u_sq, &Ls[..]);

    // Step 3 in Hyrax's Figure 8
    transcript.absorb(b"beta", &self.beta);
    transcript.absorb(b"delta", &self.delta);

    let chal = transcript.squeeze(b"chal_z")?;

    // Step 5 in Hyrax's Figure 8
    // P^(chal*a) * beta^a * delta^1
    let left_hand_side = E::GE::vartime_multiscalar_mul(
      &[(chal * a), a, E::Scalar::ONE],
      &[P_comm.affine(), self.beta.affine(), self.delta.affine()],
    );

    // g_hat^z1 * g^(a*z1) * h^z2
    let right_hand_side = E::GE::vartime_multiscalar_mul(
      &[self.z_1, (self.z_1 * a), self.z_2],
      &[
        g_hat.affine(),
        ck_y.get_ck()[0].clone(),
        ck_y.get_h().clone(),
      ],
    );

    if left_hand_side == right_hand_side {
      Ok(())
    } else {
      println!("Invalid IPA!");
      Err(NovaError::InvalidIPA)
    }
  }

  // from Spartan, notably without the zeroizing buffer
  fn batch_invert(inputs: &mut [E::Scalar]) -> E::Scalar {
    // This code is essentially identical to the FieldElement
    // implementation, and is documented there.  Unfortunately,
    // it's not easy to write it generically, since here we want
    // to use `UnpackedScalar`s internally, and `Scalar`s
    // externally, but there's no corresponding distinction for
    // field elements.
    let n = inputs.len();
    let one = E::Scalar::ONE;
    // Place scratch storage in a Zeroizing wrapper to wipe it when
    // we pass out of scope.
    let mut scratch = vec![one; n];
    //let mut scratch = Zeroizing::new(scratch_vec);
    // Keep an accumulator of all of the previous products
    let mut acc = E::Scalar::ONE;
    // Pass through the input vector, recording the previous
    // products in the scratch space
    for (input, scratch) in inputs.iter().zip(scratch.iter_mut()) {
      *scratch = acc;
      acc *= input;
    }
    // acc is nonzero iff all inputs are nonzero
    debug_assert!(acc != E::Scalar::ZERO);
    // Compute the inverse of all products
    acc = acc.invert().unwrap();
    // We need to return the product of all inverses later
    let ret = acc;
    // Pass through the vector backwards to compute the inverses
    // in place
    for (input, scratch) in inputs.iter_mut().rev().zip(scratch.iter().rev()) {
      let tmp = acc * *input;
      *input = acc * scratch;
      acc = tmp;
    }
    ret
  }

  // copied almost directly from the Spartan method, with some type massaging
  fn verification_scalars(
    &self,
    n: usize,
    transcript: &mut E::TE,
  ) -> Result<(Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
    let lg_n = self.P_L_vec.len();
    if lg_n >= 32 {
      // 4 billion multiplications should be enough for anyone
      // and this check prevents overflow in 1<<lg_n below.
      return Err(NovaError::ProofVerifyError);
    }
    if n != (1 << lg_n) {
      return Err(NovaError::ProofVerifyError);
    }

    let mut challenges = (0..lg_n)
      .map(|i| {
        transcript.absorb(b"L", &self.P_L_vec[i]);
        transcript.absorb(b"R", &self.P_R_vec[i]);
        transcript.squeeze(b"challenge_r")
      })
      .collect::<Result<Vec<E::Scalar>, NovaError>>()?;

    // inverses
    let mut challenges_inv = challenges.clone();
    let prod_all_inv = Self::batch_invert(&mut challenges_inv);
    // squares of challenges & inverses
    for i in 0..lg_n {
      challenges[i] = challenges[i].square();
      challenges_inv[i] = challenges_inv[i].square();
    }
    let challenges_sq = challenges;
    let challenges_inv_sq = challenges_inv;
    // s values inductively
    let mut s = Vec::with_capacity(n);
    s.push(prod_all_inv);
    for i in 1..n {
      let lg_i = (32 - 1 - (i as u32).leading_zeros()) as usize;
      let k = 1 << lg_i;
      // The challenges are stored in "creation order" as [u_k,...,u_1],
      // so u_{lg(i)+1} = is indexed by (lg_n-1) - lg_i
      let u_lg_i_sq = challenges_sq[(lg_n - 1) - lg_i];
      s.push(s[i - k] * u_lg_i_sq);
    }
    Ok((challenges_sq, challenges_inv_sq, s))
  }

  fn bullet_reduce_prover(
    r_P: &E::Scalar,
    x_vec: &[E::Scalar],
    a_vec: &[E::Scalar],
    y: &E::Scalar,
    gens: &CommitmentKey<E>,
    gens_y: &CommitmentKey<E>,
    transcript: &mut E::TE,
  ) -> Result<
    (
      E::Scalar,        // r_P'
      Commitment<E>,    // P_L
      Commitment<E>,    // P_R
      E::Scalar,        // y_prime
      Vec<E::Scalar>,   // x_vec'
      Vec<E::Scalar>,   // a_vec'
      CommitmentKey<E>, // gens'
    ),
    NovaError,
  > {
    let n = x_vec.len();
    let (gens_L, gens_R) = gens.split_at(n / 2);

    let y_L = Self::inner_product(&x_vec[0..n / 2], &a_vec[n / 2..n]);
    let y_R = Self::inner_product(&x_vec[n / 2..n], &a_vec[0..n / 2]);

    let r_L = E::Scalar::random(&mut OsRng);
    let r_R = E::Scalar::random(&mut OsRng);

    let P_L = CE::<E>::commit(
      &gens_R.combine(gens_y),
      &x_vec[0..n / 2]
        .iter()
        .chain(iter::once(&y_L))
        .copied()
        .collect::<Vec<E::Scalar>>(),
      &r_L,
    );

    let P_R = CE::<E>::commit(
      &gens_L.combine(gens_y),
      &x_vec[n / 2..n]
        .iter()
        .chain(iter::once(&y_R))
        .copied()
        .collect::<Vec<E::Scalar>>(),
      &r_R,
    );

    transcript.absorb(b"L", &P_L);
    transcript.absorb(b"R", &P_R);
    let chal = transcript.squeeze(b"challenge_r")?;

    let chal_square = chal * chal;
    let chal_inverse = chal.invert().unwrap();
    let chal_inverse_square = chal_inverse * chal_inverse;

    // fold the left half and the right half
    let x_vec_prime = x_vec[0..n / 2]
      .par_iter()
      .zip(x_vec[n / 2..n].par_iter())
      .map(|(x_L, x_R)| *x_L * chal + chal_inverse * *x_R)
      .collect::<Vec<E::Scalar>>();

    let a_vec_prime = a_vec[0..n / 2]
      .par_iter()
      .zip(a_vec[n / 2..n].par_iter())
      .map(|(a_L, a_R)| *a_L * chal_inverse + chal * *a_R)
      .collect::<Vec<E::Scalar>>();

    let gens_folded = gens.fold(&chal_inverse, &chal);

    let y_prime = chal_square * y_L + y + chal_inverse_square * y_R;
    let r_P_prime = r_L * chal_square + r_P + r_R * chal_inverse_square;

    Ok((
      r_P_prime,
      P_L,
      P_R,
      y_prime,
      x_vec_prime,
      a_vec_prime,
      gens_folded,
    ))
  }

  pub(crate) fn inner_product(a: &[E::Scalar], b: &[E::Scalar]) -> E::Scalar {
    assert_eq!(a.len(), b.len());
    (0..a.len())
      .into_par_iter()
      .map(|i| a[i] * b[i])
      .reduce(|| E::Scalar::ZERO, |x, y| x + y)
  }
}
