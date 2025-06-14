//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! We provide two implementations, one in snark.rs (which does not use any preprocessing)
//! and another in ppsnark.rs (which uses preprocessing to keep the verifier's state small if the PCS provides a succinct verifier)
//! We also provide direct.rs that allows proving a step circuit directly with either of the two SNARKs.
//!
//! In polynomial.rs we also provide foundational types and functions for manipulating multilinear polynomials.
pub mod direct;
// pub mod ppsnark;
pub mod snark;

#[macro_use]
mod macros;
pub(crate) mod math;
pub(crate) mod polys;
pub(crate) mod sumcheck;

use crate::{
  r1cs::{R1CSShape, SparseMatrix},
  traits::Engine,
  Commitment,
};
use ff::Field;
use itertools::Itertools as _;
use rayon::{iter::IntoParallelRefIterator, prelude::*};
use serde::{Deserialize, Serialize};

// Creates a vector of the first `n` powers of `s`.
fn powers<E: Engine>(s: &E::Scalar, n: usize) -> Vec<E::Scalar> {
  assert!(n >= 1);
  let mut powers = Vec::with_capacity(n);
  powers.push(E::Scalar::ONE);
  for i in 1..n {
    powers.push(powers[i - 1] * s);
  }
  powers
}

/// A type that holds a witness to a polynomial evaluation instance
pub struct PolyEvalWitness<E: Engine> {
  p: Vec<E::Scalar>, // polynomial
}

impl<E: Engine> PolyEvalWitness<E> {
  /// Given [Pᵢ] and s, compute P = ∑ᵢ sⁱ⋅Pᵢ
  ///
  /// # Details
  ///
  /// We allow the input polynomials to have different sizes, and interpret smaller ones as
  /// being padded with 0 to the maximum size of all polynomials.
  fn batch_diff_size(W: Vec<PolyEvalWitness<E>>, s: E::Scalar) -> PolyEvalWitness<E> {
    let powers = powers::<E>(&s, W.len());

    let size_max = W.iter().map(|w| w.p.len()).max().unwrap();
    // Scale the input polynomials by the power of s
    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = size_max / num_chunks;

    let p = if chunk_size > 0 {
      (0..num_chunks)
        .into_par_iter()
        .flat_map_iter(|chunk_index| {
          let mut chunk = vec![E::Scalar::ZERO; chunk_size];
          for (coeff, poly) in powers.iter().zip(W.iter()) {
            for (rlc, poly_eval) in chunk
              .iter_mut()
              .zip(poly.p[chunk_index * chunk_size..].iter())
            {
              if *coeff == E::Scalar::ONE {
                *rlc += *poly_eval;
              } else {
                *rlc += *coeff * poly_eval;
              };
            }
          }
          chunk
        })
        .collect::<Vec<_>>()
    } else {
      W.into_par_iter()
        .zip_eq(powers.par_iter())
        .map(|(mut w, s)| {
          if *s != E::Scalar::ONE {
            w.p.par_iter_mut().for_each(|e| *e *= s);
          }
          w.p
        })
        .reduce(
          || vec![E::Scalar::ZERO; size_max],
          |left, right| {
            // Sum into the largest polynomial
            let (mut big, small) = if left.len() > right.len() {
              (left, right)
            } else {
              (right, left)
            };

            big
              .par_iter_mut()
              .zip(small.par_iter())
              .for_each(|(b, s)| *b += s);

            big
          },
        )
    };

    PolyEvalWitness { p }
  }

  /// Given a set of polynomials \[Pᵢ\] and a scalar `s`, this method computes the weighted sum
  /// of the polynomials, where each polynomial Pᵢ is scaled by sⁱ. The method handles polynomials
  /// of different sizes by padding smaller ones with zeroes up to the size of the largest polynomial.
  ///
  /// # Panics
  ///
  /// This method panics if the polynomials in `p_vec` are not all of the same length.
  fn batch(p_vec: &[&Vec<E::Scalar>], s: &E::Scalar) -> PolyEvalWitness<E> {
    p_vec
      .iter()
      .for_each(|p| assert_eq!(p.len(), p_vec[0].len()));

    let powers_of_s = powers::<E>(s, p_vec.len());

    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = p_vec[0].len() / num_chunks;

    let p = if chunk_size > 0 {
      (0..num_chunks)
        .into_par_iter()
        .flat_map_iter(|chunk_index| {
          let mut chunk = vec![E::Scalar::ZERO; chunk_size];
          for (coeff, poly) in powers_of_s.iter().zip(p_vec.iter()) {
            for (rlc, poly_eval) in chunk
              .iter_mut()
              .zip(poly[chunk_index * chunk_size..].iter())
            {
              if *coeff == E::Scalar::ONE {
                *rlc += *poly_eval;
              } else {
                *rlc += *coeff * poly_eval;
              };
            }
          }
          chunk
        })
        .collect::<Vec<_>>()
    } else {
      zip_with!(par_iter, (p_vec, powers_of_s), |v, weight| {
        // compute the weighted sum for each vector
        v.iter().map(|&x| x * *weight).collect::<Vec<E::Scalar>>()
      })
      .reduce(
        || vec![E::Scalar::ZERO; p_vec[0].len()],
        |acc, v| {
          // perform vector addition to combine the weighted vectors
          zip_with!((acc.into_iter(), v), |x, y| x + y).collect()
        },
      )
    };

    PolyEvalWitness { p }
  }
}

/// A type that holds a polynomial evaluation instance
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PolyEvalInstance<E: Engine> {
  c: Commitment<E>,  // commitment to the polynomial
  x: Vec<E::Scalar>, // evaluation point
  e: E::Scalar,      // claimed evaluation
}

impl<E: Engine> PolyEvalInstance<E> {
  fn batch_diff_size(
    c_vec: &[Commitment<E>],
    e_vec: &[E::Scalar],
    num_vars: &[usize],
    x: Vec<E::Scalar>,
    s: E::Scalar,
  ) -> PolyEvalInstance<E> {
    let num_instances = num_vars.len();
    assert_eq!(c_vec.len(), num_instances);
    assert_eq!(e_vec.len(), num_instances);

    let num_vars_max = x.len();
    let powers: Vec<E::Scalar> = powers::<E>(&s, num_instances);
    // Rescale evaluations by the first Lagrange polynomial,
    // so that we can check its evaluation against x
    let evals_scaled = zip_with!(iter, (e_vec, num_vars), |eval, num_rounds| {
      // x_lo = [ x[0]   , ..., x[n-nᵢ-1] ]
      // x_hi = [ x[n-nᵢ], ..., x[n]      ]
      let (r_lo, _r_hi) = x.split_at(num_vars_max - num_rounds);
      // Compute L₀(x_lo)
      let lagrange_eval = r_lo
        .iter()
        .map(|r| E::Scalar::ONE - r)
        .product::<E::Scalar>();

      // vᵢ = L₀(x_lo)⋅Pᵢ(x_hi)
      lagrange_eval * eval
    })
    .collect::<Vec<_>>();

    // C = ∑ᵢ γⁱ⋅Cᵢ
    let comm_joint = zip_with!(iter, (c_vec, powers), |c, g_i| *c * *g_i)
      .fold(Commitment::<E>::default(), |acc, item| acc + item);

    // v = ∑ᵢ γⁱ⋅vᵢ
    let eval_joint = zip_with!((evals_scaled.into_iter(), powers.iter()), |e, g_i| e * g_i).sum();

    PolyEvalInstance {
      c: comm_joint,
      x,
      e: eval_joint,
    }
  }

  fn batch(
    c_vec: &[Commitment<E>],
    x: &[E::Scalar],
    e_vec: &[E::Scalar],
    s: &E::Scalar,
  ) -> PolyEvalInstance<E> {
    let num_instances = c_vec.len();
    assert_eq!(e_vec.len(), num_instances);

    let powers_of_s = powers::<E>(s, num_instances);
    // Weighted sum of evaluations
    let e = zip_with!(par_iter, (e_vec, powers_of_s), |e, p| *e * p).sum();
    // Weighted sum of commitments
    let c = zip_with!(par_iter, (c_vec, powers_of_s), |c, p| *c * *p)
      .reduce(Commitment::<E>::default, |acc, item| acc + item);

    PolyEvalInstance {
      c,
      x: x.to_vec(),
      e,
    }
  }
}

/// Bounds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
fn compute_eval_table_sparse<E: Engine>(
  S: &R1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons);
  let total_num_vars = S.num_vars;

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, ptrs) in M.indptr.windows(2).enumerate() {
      for (val, col_idx) in M.get_row_unchecked(ptrs.try_into().unwrap()) {
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * total_num_vars];
      inner(&S.A, &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * total_num_vars];
          inner(&S.B, &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * total_num_vars];
          inner(&S.C, &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}
