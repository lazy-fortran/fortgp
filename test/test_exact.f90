module exact_tests
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use fortgp, only: exact_gp_t, rbf_covariance, rbf_covariance_jvp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
contains
    subroutine collect(tests)
        type(unittest_type), allocatable, intent(out) :: tests(:)
        allocate(tests(8))
        tests(1) = new_unittest('scalar-joint-latent-and-observation', scalar)
        tests(2) = new_unittest('correlated-noise-known-answer', correlated)
        tests(3) = new_unittest('anisotropic-kernel', anisotropic)
        tests(4) = new_unittest('posterior-and-likelihood-jvp', derivatives)
        tests(5) = new_unittest('observation-vjp', observation_vjp)
        tests(6) = new_unittest('noiseless-interpolation', noiseless)
        tests(7) = new_unittest('invalid-inputs-and-failed-refit', invalid)
        tests(8) = new_unittest('prediction-after-changing-fit-size', refit)
    end subroutine collect

    subroutine scalar(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp
        type(fortnum_status_t) :: s
        real(dp), allocatable :: mean(:), latent(:, :), observed(:, :)
        real(dp) :: noise(2, 2), value
        call gp%fit(reshape([0.0_dp], [1, 1]), [3.0_dp], &
            reshape([1.0_dp], [1, 1]), 2.0_dp, [0.7_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        noise = reshape([0.4_dp, 0.1_dp, 0.1_dp, 0.9_dp], [2, 2])
        call gp%predict(reshape([0.0_dp, 0.0_dp], [1, 2]), mean, latent, s, noise, observed)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, maxval(abs(mean - 2.4_dp)) < 1e-14_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(latent - 0.8_dp)) < 1e-14_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(observed - latent - noise)) < 1e-14_dp)
        if (allocated(error)) return
        call gp%log_marginal_likelihood(value, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, abs(value + 0.5_dp*(log(10*acos(-1.0_dp)) + 1.8_dp)) < 1e-14_dp)
    end subroutine scalar

    subroutine correlated(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp
        type(fortnum_status_t) :: s
        real(dp) :: x(1, 2), noise(2, 2), kernel(2, 2), inverse(2, 2), y(2), det, value
        real(dp), allocatable :: mean(:), latent(:, :)
        real(dp) :: alpha(2), solved(2,2), expected_mean(2), expected_cov(2,2)
        x(1, :) = [0.0_dp, sqrt(2*log(2.0_dp))]
        noise = reshape([0.25_dp, 0.1_dp, 0.1_dp, 0.25_dp], [2, 2])
        kernel = reshape([1.0_dp, 0.5_dp, 0.5_dp, 1.0_dp], [2, 2])
        y = [1.0_dp, -0.5_dp]
        det = 1.25_dp**2 - 0.6_dp**2
        inverse = reshape([1.25_dp, -0.6_dp, -0.6_dp, 1.25_dp], [2, 2])/det
        call gp%fit(x, y, noise, 1.0_dp, [1.0_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%predict(x, mean, latent, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        alpha = matmul(inverse, y)
        solved = matmul(inverse, kernel)
        expected_mean = matmul(kernel, alpha)
        expected_cov = matmul(kernel, solved)
        expected_cov = kernel - expected_cov
        call check(error, maxval(abs(mean - expected_mean)) < 2e-14_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(latent - expected_cov)) < 2e-14_dp)
        if (allocated(error)) return
        call gp%log_marginal_likelihood(value, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, abs(value + log(2*acos(-1.0_dp)) + 0.5_dp*log(det) + &
            0.5_dp*dot_product(y, alpha)) < 2e-14_dp)
    end subroutine correlated

    subroutine refit(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp
        type(fortnum_status_t) :: s
        real(dp) :: x(1, 2), noise(2, 2)
        real(dp), allocatable :: mean(:), latent(:, :), observed(:, :)
        x(1, :) = [0.0_dp, 1.0_dp]
        noise = reshape([0.2_dp, 0.04_dp, 0.04_dp, 0.3_dp], [2, 2])
        call gp%fit(x, [0.3_dp, 0.8_dp], noise, 1.2_dp, [0.7_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%predict(x, mean, latent, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%fit(reshape([0.0_dp], [1, 1]), [3.0_dp], &
            reshape([1.0_dp], [1, 1]), 2.0_dp, [0.7_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%predict(reshape([0.0_dp], [1, 1]), mean, latent, s, &
            reshape([0.4_dp], [1, 1]), observed)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, abs(mean(1) - 2.4_dp) < 1e-14_dp)
        if (allocated(error)) return
        call check(error, abs(latent(1, 1) - 0.8_dp) < 1e-14_dp)
        if (allocated(error)) return
        call check(error, abs(observed(1, 1) - 1.2_dp) < 1e-14_dp)
    end subroutine refit

    subroutine anisotropic(error)
        type(error_type), allocatable, intent(out) :: error
        type(fortnum_status_t) :: s
        real(dp), allocatable :: kernel(:, :), tangent(:, :)
        real(dp) :: x(2, 1), z(2, 1), dx(2, 1), dz(2, 1), length(2), dl(2)
        real(dp) :: q, dq, expected, a, da
        x(:, 1) = [0.5_dp, -0.4_dp]; z(:, 1) = [-0.2_dp, 0.8_dp]
        dx(:, 1) = [0.2_dp, 0.1_dp]; dz(:, 1) = [-0.1_dp, 0.3_dp]
        length = [0.8_dp, 1.3_dp]; dl = [0.12_dp, -0.08_dp]
        a = 1.4_dp; da = -0.2_dp
        q = sum(((x(:, 1) - z(:, 1))/length)**2)
        dq = sum(2*(x(:, 1) - z(:, 1))*(dx(:, 1) - dz(:, 1))/length**2 &
            - 2*(x(:, 1) - z(:, 1))**2*dl/length**3)
        expected = a*a*exp(-q/2)
        call rbf_covariance(x, z, a, length, kernel, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, abs(kernel(1, 1) - expected) < 2e-15_dp)
        if (allocated(error)) return
        call rbf_covariance_jvp(x, z, a, length, dx, dz, da, dl, tangent, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, abs(tangent(1, 1) - expected*(2*da/a - dq/2)) < 2e-15_dp)
    end subroutine anisotropic

    subroutine derivatives(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp, plus, minus
        type(fortnum_status_t) :: s
        real(dp) :: x(2, 3), dx(2, 3), query(2, 2), dquery(2, 2), y(3), dy(3)
        real(dp) :: noise(3, 3), dn(3, 3), length(2), dl(2), a, da
        real(dp) :: vp, vm, dv
        real(dp) :: xp(2,3), yp(3), np(3,3), lp(2), qp(2,2)
        real(dp), parameter :: h = 2e-5_dp
        real(dp), allocatable :: dm(:), dc(:, :), mp(:), mm(:), cp(:, :), cm(:, :)
        x = reshape([0.1_dp, 0.2_dp, 0.5_dp, 0.7_dp, 1.2_dp, -0.3_dp], [2, 3])
        dx = reshape([0.05_dp, 0.02_dp, -0.01_dp, 0.03_dp, -0.04_dp, 0.02_dp], [2, 3])
        query = reshape([0.3_dp, 0.4_dp, 1.0_dp, -0.2_dp], [2, 2])
        dquery = reshape([-0.04_dp, 0.02_dp, 0.03_dp, -0.03_dp], [2, 2])
        y = [0.2_dp, 0.8_dp, -0.3_dp]; dy = [0.1_dp, -0.2_dp, 0.05_dp]
        noise = reshape([0.2_dp, 0.03_dp, 0.01_dp, 0.03_dp, 0.3_dp, 0.02_dp, &
            0.01_dp, 0.02_dp, 0.25_dp], [3, 3])
        dn = reshape([0.01_dp, -0.003_dp, 0.002_dp, -0.003_dp, -0.01_dp, 0.001_dp, &
            0.002_dp, 0.001_dp, 0.02_dp], [3, 3])
        length = [0.7_dp, 1.3_dp]; dl = [0.05_dp, -0.04_dp]
        a = 1.2_dp; da = -0.12_dp
        call gp%fit(x, y, noise, a, length, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%predict_jvp(query, dquery, dx, dy, dn, da, dl, dm, dc, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%log_marginal_likelihood_jvp(dx, dy, dn, da, dl, dv, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        xp = x + h*dx; yp = y + h*dy; np = noise + h*dn; lp = length + h*dl
        call plus%fit(xp, yp, np, a + h*da, lp, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        xp = x - h*dx; yp = y - h*dy; np = noise - h*dn; lp = length - h*dl
        call minus%fit(xp, yp, np, a - h*da, lp, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        qp = query + h*dquery
        call plus%predict(qp, mp, cp, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        qp = query - h*dquery
        call minus%predict(qp, mm, cm, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call plus%log_marginal_likelihood(vp, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call minus%log_marginal_likelihood(vm, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, maxval(abs(dm - (mp - mm)/(2*h))) < 2e-9_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(dc - (cp - cm)/(2*h))) < 2e-9_dp)
        if (allocated(error)) return
        call check(error, abs(dv - (vp - vm)/(2*h)) < 2e-9_dp)
    end subroutine derivatives

    subroutine observation_vjp(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp
        type(fortnum_status_t) :: s
        real(dp) :: x(1, 2), query(1, 3), noise(2, 2), dy(2), weights(3)
        real(dp), allocatable :: gradient(:), dm(:), dc(:, :)
        real(dp) :: zx(1,2), zq(1,3), zn(2,2)
        x(1, :) = [0.0_dp, 1.0_dp]; query(1, :) = [-0.2_dp, 0.4_dp, 1.2_dp]
        noise = reshape([0.2_dp, 0.04_dp, 0.04_dp, 0.3_dp], [2, 2])
        dy = [0.4_dp, -0.3_dp]; weights = [-0.2_dp, 0.5_dp, 0.8_dp]
        call gp%fit(x, [0.3_dp, 0.8_dp], noise, 1.2_dp, [0.7_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call gp%predict_y_vjp(query, weights, gradient, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        zx = 0.0_dp; zq = 0.0_dp; zn = 0.0_dp
        call gp%predict_jvp(query, zq, zx, dy, zn, 0.0_dp, [0.0_dp], dm, dc, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, abs(dot_product(weights, dm) - dot_product(gradient, dy)) < 1e-14_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(dc)) < 1e-14_dp)
    end subroutine observation_vjp

    subroutine noiseless(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 3), y(3), noise(3, 3)
        real(dp), allocatable :: mean(:), covariance(:, :)
        x = reshape([0.1_dp, 0.2_dp, 0.5_dp, 0.7_dp, 1.2_dp, -0.3_dp], [2, 3])
        y = [0.2_dp, 0.8_dp, -0.3_dp]
        noise = 0.0_dp
        call gp%fit(x, y, noise, 1.2_dp, [0.7_dp, 1.3_dp], status)
        call check(error, status_ok(status), status%msg)
        if (allocated(error)) return
        call gp%predict(x, mean, covariance, status)
        call check(error, status_ok(status), status%msg)
        if (allocated(error)) return
        call check(error, maxval(abs(mean - y)) < 2e-14_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(covariance)) < 2e-14_dp)
    end subroutine noiseless

    subroutine invalid(error)
        type(error_type), allocatable, intent(out) :: error
        type(exact_gp_t) :: gp
        type(fortnum_status_t) :: s
        real(dp) :: x(1, 2), noise(2, 2), nan, zx(1,2), ynan(2), qnan(1,1)
        real(dp), allocatable :: mean(:), cov(:, :), gradient(:)
        real(dp) :: weights(2)
        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        x(1, :) = [0.0_dp, 1.0_dp]
        noise = reshape([0.2_dp, 0.0_dp, 0.0_dp, 0.2_dp], [2, 2])
        call gp%predict(x, mean, cov, s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        call gp%fit(x, [0.1_dp, 0.3_dp], noise, 1.0_dp, [0.7_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        noise(1, 2) = 0.1_dp
        call gp%fit(x, [0.1_dp, 0.3_dp], noise, 1.0_dp, [0.7_dp], s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        call gp%predict(x, mean, cov, s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        noise = reshape([0.2_dp, 0.3_dp, 0.3_dp, 0.2_dp], [2, 2])
        call gp%fit(x, [0.1_dp, 0.3_dp], noise, 1.0_dp, [0.7_dp], s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        noise = 0.0_dp
        zx = 0.0_dp
        call gp%fit(zx, [0.1_dp, 0.3_dp], noise, 1.0_dp, [0.7_dp], s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        call gp%fit(x, [0.1_dp, 0.3_dp], noise, 1.0_dp, [0.0_dp], s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        ynan(1) = 0.1_dp
        ynan(2) = nan
        call gp%fit(x, ynan, noise, 1.0_dp, [0.7_dp], s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        call gp%fit(x, [0.1_dp, 0.3_dp], noise, 1.0_dp, [0.7_dp], s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        qnan = nan
        call gp%predict(qnan, mean, cov, s)
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        call gp%predict(x, mean, cov, s, &
            future_noise_cov=reshape([0.2_dp, 0.3_dp, 0.3_dp, 0.2_dp], [2, 2]))
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        weights = huge(1.0_dp)
        call gp%predict_y_vjp(x, weights, gradient, s)
        call check(error,.not. status_ok(s))
    end subroutine invalid
end module exact_tests

program test_exact
    use, intrinsic :: iso_fortran_env, only: error_unit
    use testdrive, only: run_testsuite
    use exact_tests, only: collect
    implicit none
    integer :: failures
    failures = 0
    call run_testsuite(collect, error_unit, failures)
    if (failures /= 0) error stop 1
end program test_exact
