program gaussian_update_curve
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortgp, only: exact_gp_t
    use fortprob_gaussian, only: gaussian_logpdf
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    integer, parameter :: n = 1201
    real(dp), parameter :: lower = -12.0_dp, upper = 12.0_dp
    real(dp), parameter :: step = (upper - lower)/real(n - 1, dp)
    real(dp), parameter :: z95 = 1.959963984540054_dp
    real(dp), parameter :: pi = acos(-1.0_dp)
    type(exact_gp_t) :: gp
    type(fortnum_status_t) :: status
    real(dp), allocatable :: mean(:), latent(:, :), observation(:, :)
    real(dp) :: value(n), reference(n), weights(n), prior(n), posterior(n)
    real(dp) :: future(n), log_density, evidence, quadrature_mean, quadrature_var
    real(dp) :: density_error, variance(3), center(3), half_width
    integer :: i, unit
    character(512) :: prefix
    character(18), parameter :: names(3) = &
        [character(18) :: 'prior', 'latent_posterior', 'future_observation']

    prefix = 'gaussian_update'
    if (command_argument_count() > 1) error stop 'Usage: gaussian_update_curve [PREFIX]'
    if (command_argument_count() == 1) call get_command_argument(1, prefix)

    ! Temperature anomaly in K; amplitude is a standard deviation, noise a variance.
    ! The dimensionless input is fixed at zero, with kernel length scale one.
    call gp%fit(reshape([0.0_dp], [1, 1]), [3.0_dp], &
        reshape([1.0_dp], [1, 1]), 2.0_dp, [1.0_dp], status)
    call require_success()
    ! One prediction returns latent uncertainty and adds independent future noise.
    call gp%predict(reshape([0.0_dp], [1, 1]), mean, latent, status, &
        future_noise_cov=reshape([0.5_dp], [1, 1]), observation_cov=observation)
    call require_success()
    if (abs(mean(1) - 2.4_dp) > 1.0e-12_dp) error stop 'Posterior mean mismatch'
    if (abs(latent(1, 1) - 0.8_dp) > 1.0e-12_dp) &
        error stop 'Latent variance mismatch'
    if (abs(observation(1, 1) - 1.3_dp) > 1.0e-12_dp) &
        error stop 'Future observation variance mismatch'

    do i = 1, n
        value(i) = lower + real(i - 1, dp)*step
        weights(i) = 2.0_dp
        if (mod(i, 2) == 0) weights(i) = 4.0_dp
        ! Independent scalar prior times observation likelihood, before normalization.
        reference(i) = exp(-value(i)**2/8.0_dp &
            - (3.0_dp - value(i))**2/2.0_dp)/(4.0_dp*pi)
        call gaussian_logpdf([value(i)], [0.0_dp], &
            reshape([4.0_dp], [1, 1]), log_density, status)
        call require_success()
        prior(i) = exp(log_density)
        call gaussian_logpdf([value(i)], mean, latent, log_density, status)
        call require_success()
        posterior(i) = exp(log_density)
        call gaussian_logpdf([value(i)], mean, observation, log_density, status)
        call require_success()
        future(i) = exp(log_density)
    end do
    ! Composite Simpson quadrature on [-12,12] K; posterior tails are negligible.
    weights(1) = 1.0_dp
    weights(n) = 1.0_dp
    weights = weights*step/3.0_dp
    evidence = dot_product(weights, reference)
    reference = reference/evidence
    quadrature_mean = dot_product(weights, value*reference)
    quadrature_var = dot_product(weights, (value - quadrature_mean)**2*reference)
    density_error = maxval(abs(posterior - reference))
    if (density_error > 1.0e-11_dp) error stop 'Independent density mismatch'
    if (abs(quadrature_mean - mean(1)) > 1.0e-11_dp) &
        error stop 'Independent mean mismatch'
    if (abs(quadrature_var - latent(1, 1)) > 1.0e-11_dp) &
        error stop 'Independent variance mismatch'

    open (newunit=unit, file=trim(prefix)//'_density.csv', status='replace')
    write (unit, '(a)') 'temperature_anomaly_K,prior_per_K,'// &
        'latent_posterior_per_K,future_observation_per_K,quadrature_posterior_per_K'
    do i = 1, n
        write (unit, '(es24.16,4(a,es24.16))') value(i), ',', prior(i), &
            ',', posterior(i), ',', future(i), ',', reference(i)
    end do
    close (unit)
    center = [0.0_dp, mean(1), mean(1)]
    variance = [4.0_dp, latent(1, 1), observation(1, 1)]
    open (newunit=unit, file=trim(prefix)//'_intervals.csv', status='replace')
    write (unit, '(a)') 'distribution,mean_K,variance_K2,lower95_K,upper95_K'
    do i = 1, 3
        half_width = z95*sqrt(variance(i))
        write (unit, '(a,4(a,es24.16))') trim(names(i)), ',', center(i), &
            ',', variance(i), ',', center(i) - half_width, ',', center(i) + half_width
    end do
    close (unit)
    print '(a,es14.6)', 'Maximum density difference (1/K): ', density_error
    print '(a,es24.16)', 'Quadrature evidence (1/K): ', evidence
    print '(a,es24.16)', 'Quadrature posterior mean (K): ', quadrature_mean
    print '(a,es24.16)', 'Quadrature posterior variance (K^2): ', quadrature_var

contains

    subroutine require_success()
        if (.not. status_ok(status)) error stop status%msg
    end subroutine require_success

end program gaussian_update_curve
