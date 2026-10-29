module cuda_tests
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use fortgp_cuda, only: rbf_apply_cuda, rbf_apply_cuda_jvp
    use fortgp_reductions, only: rbf_apply, rbf_apply_jvp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
contains
    subroutine collect(tests)
        type(unittest_type), allocatable, intent(out) :: tests(:)
        tests = [new_unittest('cuda-tile-tails-dense-oracle-and-jvp', kernel), &
            new_unittest('cuda-invalid-device-and-shapes', invalid)]
    end subroutine

    subroutine kernel(error)
        type(error_type), allocatable, intent(out) :: error
        type(fortnum_status_t) :: s
        real(dp) :: x(2, 131), z(2, 257), weights(257), dx(2, 131), dz(2, 257)
        real(dp) :: dw(257), ell(2), dl(2), expected(131), a, da
        real(dp), allocatable :: cpu(:), gpu(:), dcpu(:), dgpu(:), plus(:), minus(:)
        real(dp), parameter :: h = 1e-5_dp
        integer :: i, j
        do i = 1, size(x, 2)
            x(:, i) = [sin(0.1_dp*i), cos(0.03_dp*i)]
        end do
        do j = 1, size(z, 2)
            z(:, j) = [cos(0.04_dp*j), sin(0.07_dp*j)]
            weights(j) = cos(0.13_dp*j)/257
        end do
        ell = [0.7_dp, 1.3_dp]; a = 1.4_dp
        dx = 0.2_dp*x; dz = -0.1_dp*z; dw = 0.3_dp*weights
        dl = [0.1_dp, -0.2_dp]; da = -0.3_dp
        expected = 0
        do i = 1, size(x, 2)
            do j = 1, size(z, 2)
                expected(i) = expected(i) + a*a* &
                    exp(-0.5_dp*sum(((x(:, i) - z(:, j))/ell)**2))*weights(j)
            end do
        end do
        call rbf_apply(x, z, a, ell, weights, cpu, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call rbf_apply_cuda(x, z, a, ell, weights, gpu, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, maxval(abs(expected - gpu)) < 2e-14_dp)
        if (allocated(error)) return
        call check(error, maxval(abs(cpu - gpu)) < 2e-14_dp)
        if (allocated(error)) return
        call rbf_apply_jvp(x, z, a, ell, weights, dx, dz, da, dl, dw, dcpu, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call rbf_apply_cuda_jvp(x, z, a, ell, weights, dx, dz, da, dl, dw, dgpu, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, maxval(abs(dcpu - dgpu)) < 2e-14_dp)
        if (allocated(error)) return
        call rbf_apply_cuda(x + h*dx, z + h*dz, a + h*da, ell + h*dl, &
            weights + h*dw, plus, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call rbf_apply_cuda(x - h*dx, z - h*dz, a - h*da, ell - h*dl, &
            weights - h*dw, minus, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, maxval(abs(dgpu - (plus - minus)/(2*h))) < 2e-9_dp)
    end subroutine

    subroutine invalid(error)
        type(error_type), allocatable, intent(out) :: error
        type(fortnum_status_t) :: s
        real(dp), allocatable :: result(:)
        call rbf_apply_cuda(reshape([0.0_dp], [1, 1]), &
            reshape([0.0_dp], [1, 1]), 1.0_dp, [1.0_dp], [1.0_dp], result, &
            s, device=-1)
        call check(error, .not. status_ok(s) .and. .not. allocated(result))
        if (allocated(error)) return
        call rbf_apply_cuda_jvp(reshape([0.0_dp], [1, 1]), &
            reshape([0.0_dp], [1, 1]), 1.0_dp, [1.0_dp], [1.0_dp], &
            reshape([0.0_dp], [1, 1]), reshape([0.0_dp], [1, 1]), &
            0.0_dp, [0.0_dp], [0.0_dp, 1.0_dp], result, s)
        call check(error, .not. status_ok(s) .and. .not. allocated(result))
    end subroutine
end module

program test_cuda
    use testdrive, only: run_testsuite
    use cuda_tests, only: collect
    use, intrinsic :: iso_fortran_env, only: error_unit
    implicit none
    integer :: failures
    failures = 0
    call run_testsuite(collect, error_unit, failures)
    if (failures /= 0) stop 1
end program
