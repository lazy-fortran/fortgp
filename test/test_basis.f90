module basis_tests
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use fortgp_basis, only: rbf_basis_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
contains
    subroutine collect(tests)
        type(unittest_type), allocatable, intent(out) :: tests(:)
        allocate(tests(3))
        tests(1) = new_unittest('kernel-reconstruction-and-standardization', reconstruct)
        tests(2) = new_unittest('fixed-basis-query-jvp', derivative)
        tests(3) = new_unittest('duplicates-constants-and-errors', invalid)
    end subroutine
    subroutine reconstruct(error)
        type(error_type), allocatable, intent(out) :: error
        type(rbf_basis_t) :: basis, transformed
        type(fortnum_status_t) :: s
        real(dp) :: x(2, 4), y(2, 4), expected(4, 4), a, b
        real(dp), allocatable :: phi(:, :), psi(:, :)
        integer :: i, j
        x(1, :) = [-2.0_dp, -0.2_dp, 0.5_dp, 3.0_dp]
        x(2, :) = [1.0_dp, -1.0_dp, 3.0_dp, 0.0_dp]
        call basis%fit(x, 4, 0.7_dp, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call basis%evaluate(x, phi, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        do j = 1, 4
            do i = 1, 4
                a = (x(1,i)-x(1,j))/sqrt(sum((x(1,:)-sum(x(1,:))/4)**2)/4)
                b = (x(2,i)-x(2,j))/sqrt(sum((x(2,:)-sum(x(2,:))/4)**2)/4)
                expected(i,j) = exp(-(a*a+b*b)/(4*0.7_dp**2))
            end do
        end do
        do j = 1, 4
            do i = 1, 4
                expected(i,j) = expected(i,j)-sum(phi(i,:)*phi(j,:))
            end do
        end do
        call check(error, maxval(abs(expected)) < 2e-9_dp)
        if (allocated(error)) return
        y(1,:) = 4*x(1,:)+12
        y(2,:) = 0.3_dp*x(2,:)-7
        call transformed%fit(y, 4, 0.7_dp, s)
        call transformed%evaluate(y, psi, s)
        call check(error, maxval(abs(phi-psi)) < 1e-12_dp)
    end subroutine
    subroutine derivative(error)
        type(error_type), allocatable, intent(out) :: error
        type(rbf_basis_t) :: basis
        type(fortnum_status_t) :: s
        real(dp) :: x(2,4), q(2,3), dx(2,3), perturbed(2,3)
        real(dp), allocatable :: plus(:, :), minus(:, :), tangent(:, :)
        x = reshape([-2.0_dp,1.0_dp,0.0_dp,-1.0_dp,1.0_dp,3.0_dp,3.0_dp,0.0_dp],[2,4])
        q = x(:,:3)+0.2_dp
        dx = reshape([0.4_dp,-0.1_dp,0.2_dp,0.3_dp,-0.4_dp,0.6_dp],[2,3])
        call basis%fit(x, 3, 1.2_dp, s)
        call basis%evaluate_jvp(q, dx, tangent, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        perturbed = q+1e-5_dp*dx
        call basis%evaluate(perturbed, plus, s)
        perturbed = q-1e-5_dp*dx
        call basis%evaluate(perturbed, minus, s)
        call check(error, maxval(abs(tangent-(plus-minus)/2e-5_dp)) < 1e-9_dp)
    end subroutine
    subroutine invalid(error)
        type(error_type), allocatable, intent(out) :: error
        type(rbf_basis_t) :: basis
        type(fortnum_status_t) :: s
        real(dp) :: x(2,4)
        real(dp), allocatable :: phi(:, :), tangent(:, :)
        x(1,:) = [0.0_dp,1.0_dp,0.0_dp,1.0_dp]
        x(2,:) = 0.1_dp
        call basis%evaluate(x, phi, s)
        call check(error, .not. status_ok(s)); if (allocated(error)) return
        call basis%fit(x, 64, 1.0_dp, s)
        call check(error, status_ok(s), s%msg); if (allocated(error)) return
        call check(error, size(basis%centres,2) == 2 .and. .not. basis%active(2))
        if (allocated(error)) return
        call check(error, basis%centres(1,1) == -1.0_dp)
        if (allocated(error)) return
        call basis%evaluate_jvp(x, spread([0.0_dp,1.0_dp],2,4), tangent, s)
        call check(error, maxval(abs(tangent)) == 0.0_dp)
        if (allocated(error)) return
        call basis%evaluate(x(:1,:), phi, s)
        call check(error, .not. status_ok(s)); if (allocated(error)) return
        call basis%fit(x, 0, 1.0_dp, s)
        call basis%evaluate(x, phi, s)
        call check(error, .not. status_ok(s)); if (allocated(error)) return
        call basis%fit(x, 2, 0.0_dp, s)
        call check(error, .not. status_ok(s)); if (allocated(error)) return
        x(1,1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call basis%fit(x, 2, 1.0_dp, s)
        call check(error, .not. status_ok(s)); if (allocated(error)) return
        x = 2.0_dp
        call basis%fit(x, 8, 1.0_dp, s)
        call basis%evaluate(x, phi, s)
        call check(error, size(phi,2) == 1)
        if (allocated(error)) return
        call check(error, maxval(abs(phi-1/sqrt(1+basis%kernel_jitter))) < 1e-14_dp)
    end subroutine
end module
program test_basis
    use, intrinsic :: iso_fortran_env, only: error_unit
    use testdrive, only: run_testsuite
    use basis_tests, only: collect
    implicit none
    integer :: failures
    failures = 0
    call run_testsuite(collect, error_unit, failures)
    if (failures /= 0) error stop 1
end program
