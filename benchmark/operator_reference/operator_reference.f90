! Run here: fpm run -- /path/to/operator_reference.csv
! All inputs, outputs and the direction parameter are dimensionless.
program operator_reference
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortgp_reductions, only: rbf_apply, rbf_apply_jvp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    integer, parameter :: nq = 301, ns = 41
    real(dp) :: x(1, nq), z(1, ns), w(ns), dx(1, nq), dz(1, ns), dw(ns)
    real(dp) :: dense(nq), plus(nq), minus(nq), fd(nq), half_fd(nq)
    real(dp), allocatable :: applied(:), tangent(:)
    real(dp), parameter :: amplitude = 1.3_dp, length(1) = [0.55_dp]
    real(dp), parameter :: da = 0.17_dp, dl(1) = [-0.08_dp]
    real(dp) :: h, value_error, tangent_error, convergence
    character(512) :: path
    integer :: i, unit
    type(fortnum_status_t) :: status
    do i = 1, nq
        x(1, i) = -3.0_dp + 6.0_dp*real(i - 1, dp)/real(nq - 1, dp)
    end do
    do i = 1, ns
        z(1, i) = -2.5_dp + 5.0_dp*real(i - 1, dp)/real(ns - 1, dp)
        w(i) = (sin(1.7_dp*z(1, i)) + 0.3_dp*cos(3.0_dp*z(1, i)))/ns
    end do
    dx = 0.12_dp*cos(x)
    dz = -0.09_dp*sin(z + 0.4_dp)
    dw = 0.07_dp*cos(2.0_dp*z(1, :))/ns
    call rbf_apply(x, z, amplitude, length, w, applied, status)
    if (.not. status_ok(status)) error stop status%msg
    call rbf_apply_jvp(x, z, amplitude, length, w, dx, dz, da, dl, dw, &
        tangent, status)
    if (.not. status_ok(status)) error stop status%msg
    call dense_formula(x, z, amplitude, length, w, dense)
    ! Balance O(h^2) truncation against O(epsilon/h) cancellation.
    h = epsilon(1.0_dp)**(1.0_dp/3.0_dp)
    call difference(h, fd)
    call difference(0.5_dp*h, half_fd)
    value_error = maxval(abs(applied - dense))
    tangent_error = maxval(abs(tangent - fd))
    convergence = maxval(abs(fd - half_fd))
    if (.not. all(ieee_is_finite(applied))) error stop 'nonfinite apply'
    if (.not. all(ieee_is_finite(tangent))) error stop 'nonfinite tangent'
    ! Allow summation roundoff and finite-difference cancellation respectively.
    if (value_error > 100.0_dp*ns*epsilon(1.0_dp)* &
        max(1.0_dp, maxval(abs(dense)))) error stop 'dense comparison failed'
    if (tangent_error > 1.0e-8_dp*max(1.0_dp, maxval(abs(fd)))) &
        error stop 'directional finite difference failed'
    if (convergence > 1.0e-8_dp*max(1.0_dp, maxval(abs(fd)))) &
        error stop 'finite difference step check failed'
    path = 'operator_reference.csv'
    if (command_argument_count() > 0) call get_command_argument(1, path)
    open (newunit=unit, file=trim(path), status='replace', action='write')
    write (unit, '(a)') 'x,apply,dense,jvp,central_difference,apply_residual,jvp_residual'
    do i = 1, nq
        write (unit, '(es24.16,6(a,es24.16))') x(1, i), ',', applied(i), &
            ',', dense(i), ',', tangent(i), ',', fd(i), ',', applied(i) - dense(i), &
            ',', tangent(i) - fd(i)
    end do
    close (unit)
    print '(a,es14.6)', 'max absolute apply residual: ', value_error
    print '(a,es14.6)', 'max absolute JVP residual: ', tangent_error
    print '(a,es14.6)', 'max absolute FD step difference: ', convergence
    print '(a,es14.6)', 'central difference step: ', h
contains
    subroutine difference(step, result)
        real(dp), intent(in) :: step
        real(dp), intent(out) :: result(:)
        real(dp) :: q(1, nq), source(1, ns), ell(1), weights(ns)
        q = x + step*dx; source = z + step*dz
        ell = length + step*dl; weights = w + step*dw
        call dense_formula(q, source, amplitude + step*da, ell, weights, plus)
        q = x - step*dx; source = z - step*dz
        ell = length - step*dl; weights = w - step*dw
        call dense_formula(q, source, amplitude - step*da, ell, weights, minus)
        result = (plus - minus)/(2.0_dp*step)
    end subroutine

    subroutine dense_formula(query, source, a, ell, weights, result)
        real(dp), intent(in) :: query(:, :), source(:, :), a, ell(:), weights(:)
        real(dp), intent(out) :: result(:)
        real(dp) :: covariance(size(query, 2), size(source, 2))
        integer :: j, k
        ! Independent assembly avoids generated leaves and the reduction loop.
        do j = 1, size(source, 2)
            do k = 1, size(query, 2)
                covariance(k, j) = a*a*exp(-0.5_dp* &
                    sum(((query(:, k) - source(:, j))/ell)**2))
            end do
        end do
        result = matmul(covariance, weights)
    end subroutine
end program
