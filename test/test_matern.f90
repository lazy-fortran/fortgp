module matern_tests
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use fortgp, only: matern32_covariance, matern52_covariance, &
        matern32_covariance_jvp, matern52_covariance_jvp
    use fortgp_basis, only: rbf_basis_t
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortnum_cholesky, only: cholesky_factorization_t
    use omp_lib, only: omp_set_num_threads
    implicit none
contains
    subroutine collect(tests)
        type(unittest_type), allocatable, intent(out) :: tests(:)
        tests = [new_unittest('matern-reference-and-directions', kernels), &
            new_unittest('matern-basis-and-default', bases)]
    end subroutine
    subroutine covariance(kind, x, z, a, ell, k, s)
        integer, intent(in) :: kind
        real(dp), intent(in) :: x(:, :), z(:, :), a, ell(:)
        real(dp), allocatable, intent(out) :: k(:, :)
        type(fortnum_status_t), intent(out) :: s
        if (kind == 1) then
            call matern32_covariance(x, z, a, ell, k, s)
        else
            call matern52_covariance(x, z, a, ell, k, s)
        end if
    end subroutine
    subroutine kernels(error)
        type(error_type), allocatable, intent(out) :: error
        real(dp) :: x(2, 4), dx(2, 4), ell(2), dl(2), a, da, r, v, h
        real(dp), allocatable :: k(:, :), plus(:, :), minus(:, :), dk(:, :)
        real(dp), allocatable :: threaded(:, :)
        type(fortnum_status_t) :: s
        type(cholesky_factorization_t) :: factor
        integer :: kind, i, j, direction
        x = reshape([0._dp, 0._dp, 0.2_dp, 0.4_dp, 2._dp, -1._dp, &
            100._dp, 200._dp], [2, 4])
        dx = reshape([0.3_dp, 0.1_dp, -0.2_dp, 0.4_dp, 0.1_dp, -0.3_dp, &
            0._dp, 0._dp], [2, 4])
        ell = [0.7_dp, 1.3_dp]; a = 1.7_dp; h = 1e-5_dp
        do kind = 1, 2
            call omp_set_num_threads(1)
            call covariance(kind, x, x, a, ell, k, s)
            call check(error, status_ok(s), s%msg); if (allocated(error)) return
            do j = 1, 4
                do i = 1, 4
                    r = sqrt(sum(((x(:, i) - x(:, j))/ell)**2))
                    if (kind == 1) then
                        v = a*a*(1 + sqrt(3._dp)*r)*exp(-sqrt(3._dp)*r)
                    else
                        v = a*a*(1 + sqrt(5._dp)*r + 5*r*r/3)*exp(-sqrt(5._dp)*r)
                    end if
                    call check(error, abs(k(i, j) - v) < 2e-14_dp)
                    if (allocated(error)) return
                end do
            end do
            call factor%factorize(k, s)
            call check(error, status_ok(s), s%msg); if (allocated(error)) return
            call check(error, maxval(abs(k - transpose(k))) == 0._dp)
            if (allocated(error)) return
            call omp_set_num_threads(4)
            call covariance(kind, x, x, a, ell, threaded, s)
            call check(error, all(k == threaded)); if (allocated(error)) return
            do direction = 1, 3
                da = 0._dp; dl = 0._dp
                if (direction == 1) da = 0.3_dp
                if (direction == 2) dl = [0.2_dp, -0.1_dp]
                if (direction == 3) then
                    call covariance(kind, x + h*dx, x, a, ell, plus, s)
                    call covariance(kind, x - h*dx, x, a, ell, minus, s)
                else
                    call covariance(kind, x, x, a + h*da, ell + h*dl, plus, s)
                    call covariance(kind, x, x, a - h*da, ell - h*dl, minus, s)
                end if
                if (kind == 1) then
                    call matern32_covariance_jvp(x, x, a, ell, &
                        merge(dx, 0._dp, direction == 3), &
                        0*dx, da, dl, dk, s)
                else
                    call matern52_covariance_jvp(x, x, a, ell, &
                        merge(dx, 0._dp, direction == 3), &
                        0*dx, da, dl, dk, s)
                end if
                call check(error, status_ok(s), s%msg); if (allocated(error)) return
                call check(error, maxval(abs(dk - (plus - minus)/(2*h))) < 2e-9_dp)
                if (allocated(error)) return
            end do
            call covariance(kind, x, x, -a, ell, k, s)
            call check(error,.not. status_ok(s)); if (allocated(error)) return
        end do
    end subroutine
    subroutine bases(error)
        type(error_type), allocatable, intent(out) :: error
        type(rbf_basis_t) :: basis, default_basis
        type(fortnum_status_t) :: s
        real(dp) :: x(1, 4), dx(1, 4), z(1, 4), ell(1)
        real(dp), allocatable :: p(:, :), plus(:, :), minus(:, :), tangent(:, :)
        real(dp), allocatable :: k(:, :)
        character(8), parameter :: names(3) = ['rbf     ', 'matern32', 'matern52']
        integer :: kind
        x(1, :) = [-1._dp, 0._dp, 0.4_dp, 2._dp]; dx = 0.3_dp
        do kind = 1, 3
            call omp_set_num_threads(1)
            call basis%fit(x, 4, 0.8_dp, s, kernel=trim(names(kind)))
            call check(error, status_ok(s), s%msg); if (allocated(error)) return
            call check(error, basis%kernel_name() == names(kind))
            if (allocated(error)) return
            call basis%evaluate(x, p, s)
            call check(error, status_ok(s), s%msg); if (allocated(error)) return
            call omp_set_num_threads(4)
            call default_basis%fit(x, 4, 0.8_dp, s, kernel=trim(names(kind)))
            call default_basis%evaluate(x, plus, s)
            call check(error, status_ok(s), s%msg); if (allocated(error)) return
            call check(error, all(p == plus)); if (allocated(error)) return
            if (kind > 1) then
                z(1, :) = (x(1, :) - sum(x)/4)/sqrt(sum((x - sum(x)/4)**2)/4)
                ell = 0.8_dp
                call covariance(kind - 1, z, z, 1._dp, ell, k, s)
                call check(error, maxval(abs(matmul(p, transpose(p)) - k)) < 2e-9_dp)
                if (allocated(error)) return
            else
                call default_basis%fit(x, 4, 0.8_dp, s)
                call default_basis%evaluate(x, plus, s)
                call check(error, all(p == plus)); if (allocated(error)) return
            end if
            call basis%evaluate_jvp(x, dx, tangent, s)
            call check(error, status_ok(s), s%msg); if (allocated(error)) return
            call basis%evaluate(x + 1e-5_dp*dx, plus, s)
            call basis%evaluate(x - 1e-5_dp*dx, minus, s)
            call check(error, maxval(abs(tangent - (plus - minus)/2e-5_dp)) < 2e-8_dp)
            if (allocated(error)) return
        end do
        call basis%fit(x, 4, 1._dp, s, kernel='invalid')
        call check(error,.not. status_ok(s)); if (allocated(error)) return
        call basis%evaluate(x, p, s)
        call check(error,.not. status_ok(s))
    end subroutine
end module
program test_matern
    use, intrinsic :: iso_fortran_env, only: error_unit
    use testdrive, only: run_testsuite
    use matern_tests, only: collect
    implicit none
    integer :: failures
    failures = 0
    call run_testsuite(collect, error_unit, failures)
    if (failures /= 0) error stop 1
end program
