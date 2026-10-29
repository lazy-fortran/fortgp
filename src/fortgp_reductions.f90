module fortgp_reductions
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private
    public :: rbf_apply, rbf_apply_jvp, rbf_apply_valid

    interface
        pure subroutine fortgp_distance_leaf(x, y, length, value, dx, dy, dl)
            import dp
            real(dp), intent(in) :: x, y, length
            real(dp), intent(out) :: value, dx, dy, dl
        end subroutine
        pure subroutine fortgp_rbf_leaf(amplitude, distance, value, da, dq)
            import dp
            real(dp), intent(in) :: amplitude, distance
            real(dp), intent(out) :: value, da, dq
        end subroutine
    end interface
contains
    logical function rbf_apply_valid(x, z, amplitude, length, weights, &
            status) result(valid)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:), weights(:)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            'fortgp reduction: incompatible or empty input shapes')
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(z, 2) < 1) return
        if (size(z, 1) /= size(x, 1)) return
        if (size(length) /= size(x, 1)) return
        if (size(weights) /= size(z, 2)) return
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            'fortgp reduction: finite inputs and positive scales required')
        if (.not. all(ieee_is_finite(x))) return
        if (.not. all(ieee_is_finite(z))) return
        if (.not. all(ieee_is_finite(length))) return
        if (.not. all(ieee_is_finite(weights))) return
        if (.not. ieee_is_finite(amplitude)) return
        if (amplitude <= 0.0_dp .or. any(length <= 0.0_dp)) return
        call status_set(status, FORTNUM_OK, '')
        valid = .true.
    end function

    subroutine rbf_apply(x, z, amplitude, length, weights, result, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:), weights(:)
        real(dp), allocatable, intent(out) :: result(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, term, gx, gz, gl, value, ga, gq, accum
        integer :: i, j, d
        if (.not. rbf_apply_valid(x, z, amplitude, length, weights, status)) return
        allocate(result(size(x, 2)))
        !$omp parallel do schedule(static) &
        !$omp private(j,d,distance,term,gx,gz,gl,value,ga,gq,accum)
        do i = 1, size(x, 2)
            accum = 0.0_dp
            do j = 1, size(z, 2)
                distance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, gx, gz, gl)
                    distance = distance + term
                end do
                call fortgp_rbf_leaf(amplitude, distance, value, ga, gq)
                accum = accum + value*weights(j)
            end do
            result(i) = accum
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(result))) then
            deallocate(result)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                'fortgp reduction: evaluation overflow or invalid arithmetic')
        end if
    end subroutine

    subroutine rbf_apply_jvp(x, z, amplitude, length, weights, dx, dz, &
            damplitude, dlength, dweights, tangent, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:), weights(:)
        real(dp), intent(in) :: dx(:, :), dz(:, :), damplitude, dlength(:), dweights(:)
        real(dp), allocatable, intent(out) :: tangent(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, dq, term, gx, gz, gl, value, ga, gq, accum
        integer :: i, j, d
        if (.not. rbf_apply_valid(x, z, amplitude, length, weights, status)) return
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            'fortgp reduction: tangent shapes must match primal shapes')
        if (any(shape(dx) /= shape(x)) .or. any(shape(dz) /= shape(z))) return
        if (size(dlength) /= size(length) .or. size(dweights) /= size(weights)) return
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            'fortgp reduction: finite tangents required')
        if (.not. all(ieee_is_finite(dx))) return
        if (.not. all(ieee_is_finite(dz))) return
        if (.not. all(ieee_is_finite(dlength))) return
        if (.not. all(ieee_is_finite(dweights))) return
        if (.not. ieee_is_finite(damplitude)) return
        allocate(tangent(size(x, 2)))
        !$omp parallel do schedule(static) &
        !$omp private(j,d,distance,dq,term,gx,gz,gl,value,ga,gq,accum)
        do i = 1, size(x, 2)
            accum = 0.0_dp
            do j = 1, size(z, 2)
                distance = 0.0_dp
                dq = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, gx, gz, gl)
                    distance = distance + term
                    dq = dq + gx*dx(d, i) + gz*dz(d, j) + gl*dlength(d)
                end do
                call fortgp_rbf_leaf(amplitude, distance, value, ga, gq)
                accum = accum + (ga*damplitude + gq*dq)*weights(j) &
                    + value*dweights(j)
            end do
            tangent(i) = accum
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(tangent))) then
            deallocate(tangent)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                'fortgp reduction: tangent evaluation overflow')
            return
        end if
        call status_set(status, FORTNUM_OK, '')
    end subroutine
end module
