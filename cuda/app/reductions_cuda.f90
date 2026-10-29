program reductions_cuda
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: iso_c_binding, only: c_size_t
    use omp_lib, only: omp_get_wtime
    use fortgp_cuda, only: rbf_apply_cuda, rbf_apply_cuda_jvp
    use fortgp_reductions, only: rbf_apply, rbf_apply_jvp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    real(dp), allocatable :: x(:, :), z(:, :), w(:), dx(:, :), dz(:, :), dw(:)
    real(dp), allocatable :: result(:), reference(:), tangent(:), tangent_ref(:)
    real(dp) :: started, elapsed, best, kernel_ms, best_kernel, error
    integer(c_size_t) :: bytes
    integer :: n, m, i, d, repeat
    character(32) :: arg
    type(fortnum_status_t) :: status
    n = 4096; m = 4096
    call get_command_argument(1, arg)
    if (len_trim(arg) > 0) read (arg, *) n
    call get_command_argument(2, arg)
    if (len_trim(arg) > 0) read (arg, *) m
    allocate(x(3, n), z(3, m), w(m), dx(3, n), dz(3, m), dw(m))
    do i = 1, n
        do d = 1, 3
            x(d, i) = sin(real(i*d, dp)*0.017_dp)
        end do
    end do
    do i = 1, m
        do d = 1, 3
            z(d, i) = cos(real(i*d, dp)*0.013_dp)
        end do
        w(i) = sin(real(i, dp)*0.031_dp)/m
    end do
    dx = 0.1_dp*x; dz = -0.2_dp*z; dw = -0.1_dp*w
    call rbf_apply(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], w, reference, status)
    if (.not. status_ok(status)) error stop status%msg
    call rbf_apply_jvp(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], w, &
        dx, dz, 0.2_dp, [0.1_dp, -0.1_dp, 0.2_dp], dw, tangent_ref, status)
    if (.not. status_ok(status)) error stop status%msg
    print '(a)', 'operation,n,m,end_to_end_seconds,kernel_seconds,device_array_bytes,max_abs_error'
    do d = 1, 2
        best = huge(1.0_dp); best_kernel = huge(1.0_dp)
        do repeat = 0, 3
            started = omp_get_wtime()
            if (d == 1) then
                call rbf_apply_cuda(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], &
                    w, result, status, device_bytes=bytes, kernel_ms=kernel_ms)
            else
                call rbf_apply_cuda_jvp(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], &
                    w, dx, dz, 0.2_dp, [0.1_dp, -0.1_dp, 0.2_dp], dw, &
                    tangent, status, device_bytes=bytes, kernel_ms=kernel_ms)
            end if
            elapsed = omp_get_wtime() - started
            if (.not. status_ok(status)) error stop status%msg
            if (repeat == 0) cycle
            best = min(best, elapsed)
            best_kernel = min(best_kernel, kernel_ms*1e-3_dp)
        end do
        if (d == 1) then
            error = maxval(abs(reference - result))
        else
            error = maxval(abs(tangent_ref - tangent))
        end if
        print '(a,i0,a,i0,2(a,es17.9),a,i0,a,es17.9)', &
            merge('apply,', 'jvp,  ', d == 1), n, ',', m, ',', best, &
            ',', best_kernel, ',', bytes, ',', error
    end do
end program
