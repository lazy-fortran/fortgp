module fortgp_cuda
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: iso_c_binding, only: c_int, c_double, c_size_t, c_ptr, &
        c_loc, c_null_ptr
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortgp_reductions, only: rbf_apply_valid
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private
    public :: rbf_apply_cuda, rbf_apply_cuda_jvp
    interface
        integer(c_int) function backend(d, n, m, a, x, z, ell, v, dx, dz, &
                da, dl, dv, result, derivative, device, bytes, ms) &
                bind(c, name='fortgp_cuda_apply')
            import c_int, c_double, c_size_t, c_ptr
            integer(c_int), value :: d, n, m, derivative, device
            real(c_double), value :: a, da
            type(c_ptr), value :: x, z, ell, v, dx, dz, dl, dv, result
            integer(c_size_t), intent(out) :: bytes
            real(c_double), intent(out) :: ms
        end function
    end interface
contains
    subroutine finish(code, result, status)
        integer(c_int), intent(in) :: code
        real(dp), allocatable, intent(inout) :: result(:)
        type(fortnum_status_t), intent(out) :: status
        character(100) :: message
        if (code /= 0) then
            deallocate(result)
            write (message, '(a,i0)') 'fortgp CUDA runtime error code: ', code
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(message))
        else if (.not. all(ieee_is_finite(result))) then
            deallocate(result)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                'fortgp CUDA: evaluation overflow or invalid arithmetic')
        else
            call status_set(status, FORTNUM_OK, '')
        end if
    end subroutine

    subroutine rbf_apply_cuda(x, z, amplitude, length, weights, result, &
            status, device, device_bytes, kernel_ms)
        real(dp), contiguous, target, intent(in) :: x(:, :), z(:, :), &
            length(:), weights(:)
        real(dp), intent(in) :: amplitude
        real(dp), allocatable, target, intent(out) :: result(:)
        type(fortnum_status_t), intent(out) :: status
        integer, optional, intent(in) :: device
        integer(c_size_t), optional, intent(out) :: device_bytes
        real(dp), optional, intent(out) :: kernel_ms
        integer(c_int) :: code, selected
        integer(c_size_t) :: bytes
        real(c_double) :: ms
        if (present(device_bytes)) device_bytes = 0
        if (present(kernel_ms)) kernel_ms = 0
        if (.not. rbf_apply_valid(x, z, amplitude, length, weights, status)) return
        selected = 0
        if (present(device)) selected = int(device, c_int)
        allocate(result(size(x, 2)))
        code = backend(int(size(x, 1), c_int), int(size(x, 2), c_int), &
            int(size(z, 2), c_int), amplitude, c_loc(x), c_loc(z), &
            c_loc(length), c_loc(weights), c_null_ptr, c_null_ptr, 0.0_dp, &
            c_null_ptr, c_null_ptr, c_loc(result), 0_c_int, selected, bytes, ms)
        call finish(code, result, status)
        if (present(device_bytes)) device_bytes = bytes
        if (present(kernel_ms)) kernel_ms = ms
    end subroutine

    subroutine rbf_apply_cuda_jvp(x, z, amplitude, length, weights, dx, dz, &
            damplitude, dlength, dweights, tangent, status, device, &
            device_bytes, kernel_ms)
        real(dp), contiguous, target, intent(in) :: x(:, :), z(:, :), &
            length(:), weights(:), dx(:, :), dz(:, :), dlength(:), dweights(:)
        real(dp), intent(in) :: amplitude, damplitude
        real(dp), allocatable, target, intent(out) :: tangent(:)
        type(fortnum_status_t), intent(out) :: status
        integer, optional, intent(in) :: device
        integer(c_size_t), optional, intent(out) :: device_bytes
        real(dp), optional, intent(out) :: kernel_ms
        integer(c_int) :: code, selected
        integer(c_size_t) :: bytes
        real(c_double) :: ms
        if (present(device_bytes)) device_bytes = 0
        if (present(kernel_ms)) kernel_ms = 0
        if (.not. rbf_apply_valid(x, z, amplitude, length, weights, status)) return
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            'fortgp CUDA: tangent shapes must match primal shapes')
        if (any(shape(dx) /= shape(x)) .or. any(shape(dz) /= shape(z))) return
        if (size(dlength) /= size(length) .or. size(dweights) /= size(weights)) return
        call status_set(status, FORTNUM_DOMAIN_ERROR, 'fortgp CUDA: finite tangents required')
        if (.not. all(ieee_is_finite(dx))) return
        if (.not. all(ieee_is_finite(dz))) return
        if (.not. all(ieee_is_finite(dlength))) return
        if (.not. all(ieee_is_finite(dweights))) return
        if (.not. ieee_is_finite(damplitude)) return
        selected = 0
        if (present(device)) selected = int(device, c_int)
        allocate(tangent(size(x, 2)))
        code = backend(int(size(x, 1), c_int), int(size(x, 2), c_int), &
            int(size(z, 2), c_int), amplitude, c_loc(x), c_loc(z), &
            c_loc(length), c_loc(weights), c_loc(dx), c_loc(dz), damplitude, &
            c_loc(dlength), c_loc(dweights), c_loc(tangent), &
            1_c_int, selected, bytes, ms)
        call finish(code, tangent, status)
        if (present(device_bytes)) device_bytes = bytes
        if (present(kernel_ms)) kernel_ms = ms
    end subroutine
end module
