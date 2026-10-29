module reduction_tests
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use fortgp, only: rbf_covariance
    use fortgp_reductions, only: rbf_apply, rbf_apply_jvp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
contains
    subroutine collect(tests)
        type(unittest_type), allocatable, intent(out) :: tests(:)
        allocate(tests(3))
        tests(1) = new_unittest('reduction-dense-and-known-answer', known)
        tests(2) = new_unittest('reduction-all-directional-derivatives', derivative)
        tests(3) = new_unittest('reduction-invalid-inputs', invalid)
    end subroutine

    subroutine known(error)
        type(error_type), allocatable, intent(out) :: error
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 3), z(2, 5), ell(2), v(5), reference(3), dense_result(3), a
        real(dp), allocatable :: result(:), matrix(:, :)
        integer :: i, j
        x = reshape([0.1_dp, -0.4_dp, 0.5_dp, 0.2_dp, 1.3_dp, -0.1_dp], [2, 3])
        z = reshape([0.2_dp, 0.3_dp, -0.4_dp, 0.9_dp, 0.1_dp, 0.6_dp, &
            0.7_dp, 0.4_dp, -0.8_dp, 0.2_dp], [2, 5])
        ell = [0.7_dp, 1.2_dp]; a = 1.4_dp
        v = [0.3_dp, -0.9_dp, 0.7_dp, 0.2_dp, -0.4_dp]
        reference = 0
        do i = 1, 3
            do j = 1, 5
                reference(i) = reference(i) + &
                    a*a*exp(-0.5_dp*sum(((x(:, i) - z(:, j))/ell)**2))*v(j)
            end do
        end do
        call rbf_apply(x, z, a, ell, v, result, status)
        call check(error, status_ok(status), status%msg)
        if (allocated(error)) return
        call check(error, maxval(abs(result - reference)) < 3e-15_dp)
        if (allocated(error)) return
        call rbf_covariance(x, z, a, ell, matrix, status)
        call check(error, status_ok(status), status%msg)
        if (allocated(error)) return
        dense_result = matmul(matrix, v)
        call check(error, maxval(abs(result - dense_result)) < 3e-15_dp)
        if (allocated(error)) return
        call rbf_apply(reshape([0.0_dp, sqrt(2*log(2.0_dp))], [1, 2]), &
            reshape([0.0_dp, sqrt(2*log(2.0_dp))], [1, 2]), &
            2.0_dp, [1.0_dp], [1.0_dp, -2.0_dp], result, status)
        call check(error, status_ok(status), status%msg)
        if (allocated(error)) return
        call check(error, maxval(abs(result - [0.0_dp, -6.0_dp])) < 3e-15_dp)
    end subroutine

    subroutine derivative(error)
        type(error_type), allocatable, intent(out) :: error
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 3), z(2, 2), dx(2, 3), dz(2, 2), dl(2), dv(2), da
        real(dp) :: all_dx(2, 3), all_dz(2, 2), ell(2), v(2)
        real(dp) :: perturbed_x(2, 3), perturbed_z(2, 2), pe(2), pv(2)
        real(dp), parameter :: h = 1e-5_dp, a = 1.3_dp
        real(dp), allocatable :: tangent(:), plus(:), minus(:)
        integer :: direction
        x = reshape([0.1_dp, 0.4_dp, -0.3_dp, 0.6_dp, 0.9_dp, -0.2_dp], [2, 3])
        z = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.3_dp], [2, 2])
        all_dx = reshape([0.2_dp, -0.1_dp, 0.3_dp, 0.1_dp, -0.2_dp, 0.4_dp], [2, 3])
        all_dz = reshape([0.1_dp, 0.2_dp, -0.2_dp, 0.3_dp], [2, 2])
        ell = [0.8_dp, 1.4_dp]; v = [0.7_dp, -0.4_dp]
        do direction = 1, 6
            dx = 0; dz = 0; dl = 0; dv = 0; da = 0
            if (direction == 1 .or. direction == 6) dx = all_dx
            if (direction == 2 .or. direction == 6) dz = all_dz
            if (direction == 3 .or. direction == 6) dl = [0.1_dp, -0.2_dp]
            if (direction == 4 .or. direction == 6) dv = [-0.2_dp, 0.3_dp]
            if (direction == 5 .or. direction == 6) da = 0.3_dp
            call rbf_apply_jvp(x, z, a, ell, v, dx, dz, da, dl, dv, tangent, status)
            call check(error, status_ok(status), status%msg)
            if (allocated(error)) return
            perturbed_x = x + h*dx; perturbed_z = z + h*dz
            pe = ell + h*dl; pv = v + h*dv
            call rbf_apply(perturbed_x, perturbed_z, a + h*da, pe, pv, plus, status)
            call check(error, status_ok(status), status%msg)
            if (allocated(error)) return
            perturbed_x = x - h*dx; perturbed_z = z - h*dz
            pe = ell - h*dl; pv = v - h*dv
            call rbf_apply(perturbed_x, perturbed_z, a - h*da, pe, pv, minus, status)
            call check(error, status_ok(status), status%msg)
            if (allocated(error)) return
            call check(error, maxval(abs(tangent - (plus - minus)/(2*h))) < 2e-9_dp)
            if (allocated(error)) return
        end do
    end subroutine

    subroutine invalid(error)
        type(error_type), allocatable, intent(out) :: error
        type(fortnum_status_t) :: status
        real(dp), allocatable :: result(:)
        real(dp) :: nan, nonfinite(1)
        nan = ieee_value(0.0_dp, ieee_quiet_nan)
        nonfinite(1) = nan
        call rbf_apply(reshape([0.0_dp], [1, 1]), reshape([0.0_dp], [1, 1]), &
            1.0_dp, [0.0_dp], [1.0_dp], result, status)
        call check(error, .not. status_ok(status) .and. .not. allocated(result))
        if (allocated(error)) return
        call rbf_apply(reshape([0.0_dp], [1, 1]), reshape([0.0_dp], [1, 1]), &
            1.0_dp, [1.0_dp], nonfinite, result, status)
        call check(error, .not. status_ok(status) .and. .not. allocated(result))
        if (allocated(error)) return
        call rbf_apply(reshape([0.0_dp], [1, 1]), reshape([0.0_dp], [1, 1]), &
            1.0_dp, [1.0_dp], [1.0_dp, 2.0_dp], result, status)
        call check(error, .not. status_ok(status) .and. .not. allocated(result))
        if (allocated(error)) return
        call rbf_apply_jvp(reshape([0.0_dp], [1, 1]), &
            reshape([0.0_dp], [1, 1]), 1.0_dp, [1.0_dp], [1.0_dp], &
            reshape([0.0_dp, 0.0_dp], [2, 1]), reshape([0.0_dp], [1, 1]), &
            0.0_dp, [0.0_dp], [0.0_dp], result, status)
        call check(error, .not. status_ok(status) .and. .not. allocated(result))
    end subroutine
end module

program test_reductions
    use testdrive, only: run_testsuite
    use reduction_tests, only: collect
    use, intrinsic :: iso_fortran_env, only: error_unit
    implicit none
    integer :: failures
    failures = 0
    call run_testsuite(collect, error_unit, failures)
    if (failures /= 0) stop 1
end program
