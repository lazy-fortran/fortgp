program scalar_gp
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortgp, only: exact_gp_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    type(exact_gp_t) :: gp
    type(fortnum_status_t) :: status
    real(dp), allocatable :: mean(:), latent_cov(:, :), observation_cov(:, :)
    call gp%fit(reshape([0.0_dp], [1, 1]), [3.0_dp], &
        reshape([1.0_dp], [1, 1]), 2.0_dp, [0.7_dp], status)
    if (.not. status_ok(status)) error stop status%msg
    call gp%predict(reshape([0.0_dp], [1, 1]), mean, latent_cov, status, &
        future_noise_cov=reshape([0.5_dp], [1, 1]), observation_cov=observation_cov)
    if (.not. status_ok(status)) error stop status%msg
    print '(a,f8.4)', 'Posterior mean: ', mean(1)
    print '(a,f8.4)', 'Latent variance: ', latent_cov(1, 1)
    print '(a,f8.4)', 'Future observation variance: ', observation_cov(1, 1)
end program scalar_gp
