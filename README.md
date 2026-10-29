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
