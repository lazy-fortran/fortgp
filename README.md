# fortgp

[fortgp](https://github.com/lazy-fortran/fortgp) is an initial implementation of exact, zero-mean scalar
Gaussian-process regression in Fortran. Training and query inputs have shape
`(ndim,npoints)`. The squared-exponential covariance has amplitude measured as a
standard deviation and one positive length per input dimension, in that
input's units. Training noise is a full covariance matrix in squared output
units.

```fortran
use fortgp, only: exact_gp_t
use fortnum_status, only: fortnum_status_t, status_ok
use, intrinsic :: iso_fortran_env, only: dp => real64
implicit none
type(exact_gp_t) :: gp
type(fortnum_status_t) :: status
real(dp), allocatable :: mean(:), latent_cov(:,:), observation_cov(:,:)

call gp%fit(reshape([0.0_dp], [1,1]), [3.0_dp], &
    reshape([1.0_dp], [1,1]), 2.0_dp, [0.7_dp], status)
if (.not. status_ok(status)) error stop status%msg
call gp%predict(reshape([0.0_dp], [1,1]), mean, latent_cov, status, &
    future_noise_cov=reshape([0.5_dp], [1,1]), observation_cov=observation_cov)
if (.not. status_ok(status)) error stop status%msg
! mean=2.4, latent_cov=0.8, observation_cov=1.3
```

`predict` returns the joint conditional covariance of the latent function,
including correlations between query points. Optional `observation_cov` adds
`future_noise_cov`, which represents noise independent of the training data
and latent process. If future noise is omitted, observation covariance equals
latent covariance. Correlation between future and training observation noise
requires a different joint model and is outside this interface.

`fit` rejects nonfinite data, nonpositive amplitudes or lengths, asymmetric or
indefinite noise covariance, and a training covariance that cannot be
factorized as positive definite. Zero noise is allowed when the resulting
training covariance is positive definite. Covariance validation permits only
roundoff-scale symmetry/eigenvalue discrepancies as documented in
[fortprob](https://github.com/lazy-fortran/fortprob); it does not clip eigenvalues.
There is no hidden jitter or variance clipping. A failed refit invalidates the
previous fit. Callers must inspect the returned `fortnum_status_t` before using
results.

`log_marginal_likelihood(value,status)` returns the normalized log density of
the training values under the fixed kernel and supplied noise. Hyperparameter
optimization and hyperparameter uncertainty are not included.

`predict_jvp(query,dquery,dx,dy,dnoise,damplitude,dlength,dmean,dlatent_cov,status)`
returns a directional derivative of the posterior mean and covariance. Here
`dx` is the training-input direction, `dy` the training-value direction,
`dquery` the query-input direction, and the other arguments are directions of
the corresponding fitted quantities. Directions have the same shapes as their
primal values; the noise direction is symmetric and may be indefinite.
`log_marginal_likelihood_jvp(dx,dy,dnoise,damplitude,dlength,dvalue,status)`
supports the same training directions. These products differentiate the conditional calculation at supplied
hyperparameters, including specified amplitude and length perturbations and
the linear solves.
They do not differentiate a fitted hyperparameter optimum.

`predict_y_vjp(query,mean_weight,y_gradient,status)` returns the transpose
product for training values and the posterior mean. The latent covariance is
independent of the training values. Transpose products for other parameters,
joint sampling, approximate inference, and GPU execution are not provided by
this initial interface. Input uncertainty and parameter uncertainty require a
specified joint covariance and a separate propagation calculation; they are
not automatically added to the conditional latent covariance.

`rbf_covariance` and `rbf_covariance_jvp` expose kernel matrices and directional
products with the same conventions. Independent pairwise assembly uses OpenMP.
This implementation assembles dense matrices and repeats the shared
conditioning factorization for predictions; it targets small exact reference
problems.

Run `fpm test` for independent scalar/correlated-noise known answers, derivative
finite differences and invalid-input checks. The normal library build needs
no computer algebra runtime. Dependencies are pinned: fortgp uses
[fortprob](https://github.com/lazy-fortran/fortprob), which uses
[fortnum](https://github.com/lazy-fortran/fortnum) for numerical primitives.

Pointwise covariance and derivative leaves are generated from the specification
in [gen/generate.f90](gen/generate.f90) using
[FortSym](https://github.com/lazy-fortran/fortsym), and checked into
`src/generated/`. Regeneration uses `cd gen && fpm run generate` with FortSym's
system prerequisites and compiler/linker include paths configured. Alternatively,
build the pinned FortSym revision with CMake and
`-DFORTSYM_USE_SYSTEM_DEPS=OFF`, compile `gen/generate.f90` against its module and
library directories, and run the generator from `gen/`. Regenerating must leave
the checked-in sources unchanged. Generated derivatives have independent
known-answer and finite-difference checks; generation itself is not a proof.

## Finite-rank kernel basis

`use fortgp_basis, only: rbf_basis_t` provides a Nyström feature map for a
finite-rank Gaussian process (GP). It can be passed to Bayesian coefficient
inference in FortProb. It does not implement a full GP or add a residual
variance correction.

```fortran
type(rbf_basis_t) :: basis
real(dp), allocatable :: phi(:,:), dphi(:,:)
call basis%fit(x, ncentres=32, length_multiplier=1.0_dp, status=status)
call basis%evaluate(query, phi, status)
call basis%evaluate_jvp(query, dquery, dphi, status)
```

The default kernel is the squared exponential (`'rbf'`). Pass the optional
`kernel='matern32'` or `kernel='matern52'` to `fit` for Matérn 3/2 or 5/2;
`basis%kernel_name()` reports the fitted choice. The type name is retained for
compatibility. The public `matern32_covariance` and `matern52_covariance`, and
their `_jvp` counterparts, have the same arguments as the RBF covariance
routines. The existing `exact_gp_t` continues to use the RBF kernel.

Check `status_ok(status)` after every call. Inputs have shape `(ndim,npoints)`;
outputs have shape `(npoints,actual_rank)`. Fit uses inputs alone, with population
means and standard deviations estimated only on its fitting inputs. Dimensions
constant during fitting are ignored in fitting, query evaluation and query
derivatives. Fit chooses the point closest to the standardized origin first,
then greedily the farthest point from the selected set; ties use the first input
index. Duplicate centres are stopped, so actual rank can be below the requested
positive count. Requests larger than the input count are allowed.

The RBF base covariance is `exp(-sum(delta**2)/(2*d_active*ell**2))`, where `delta`
is the standardized input difference, `d_active` counts nonconstant dimensions
(with a minimum of one), and `ell` is the positive `length_multiplier`.
Writing `r=sqrt(sum(delta**2)/d_active)/ell`, the Matérn alternatives use
`(1+sqrt(3)*r)*exp(-sqrt(3)*r)` and
`(1+sqrt(5)*r+5*r**2/3)*exp(-sqrt(5)*r)`. Covariance amplitudes are one in
the basis; coefficient priors set the model's output scale. Generated
derivatives use finite limits at coincident inputs.
Here `K_mm` is the centre covariance, `K_nm` is the query-to-centre
covariance and `I` is the identity matrix. With the lower Cholesky factor `L`
defined by `K_mm + jitter*I = L*transpose(L)`, features are
`K_nm*inverse(transpose(L))`.
`kernel_jitter` defaults to `1e-10` and may be set to another positive finite
value before fitting. It is a declared numerical stabilization, not observation
noise or a fitted statistical parameter. There is no adaptive jitter retry.

The public `means`, `scales`, `active` and standardized `centres` record the
fitted transform; treat these fields and `length_multiplier` as read-only after
fit. `clear()` invalidates the fit. Any failed refit also invalidates it.
`evaluate_jvp` is a Jacobian-vector product for query inputs with the fitted
transform, centre selection and factor held fixed. It does not differentiate
through fitting or discrete centre selection.

Centre selection uses linear storage in the fitting count and `O(npoints *
ncentres * ndim)` work; no fitting-count-squared covariance is formed. Evaluation
uses rectangular kernel storage. Estimated workspace requests exceeding four GiB are
rejected before allocation. The selection-distance and covariance kernels use
OpenMP. Run the synthetic benchmark with
`cd benchmark/basis && fpm run --profile release basis`, with the linear algebra
library's thread count fixed to one, to compare one and four OpenMP threads
and numerical agreement.
