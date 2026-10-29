program reductions
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use omp_lib, only: omp_set_num_threads, omp_get_wtime, omp_get_max_threads
    use fortgp_reductions, only: rbf_apply, rbf_apply_jvp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    real(dp), allocatable :: x(:, :), z(:, :), weights(:), result(:), reference(:)
    real(dp), allocatable :: dx(:, :), dz(:, :), dw(:), tangent(:), tangent_ref(:)
    real(dp) :: elapsed, started, error, derror
    integer :: n, m, i, d, threads, available, repeat
    character(32) :: arg
    type(fortnum_status_t) :: status
    n = 4096; m = 4096
    call get_command_argument(1, arg)
    if (len_trim(arg) > 0) read (arg, *) n
    call get_command_argument(2, arg)
    if (len_trim(arg) > 0) read (arg, *) m
    allocate(x(3, n), z(3, m), weights(m), dx(3, n), dz(3, m), dw(m))
    do i = 1, n
        do d = 1, 3
            x(d, i) = sin(real(i*d, dp)*0.017_dp)
        end do
    end do
    do i = 1, m
        do d = 1, 3
            z(d, i) = cos(real(i*d, dp)*0.013_dp)
        end do
        weights(i) = sin(real(i, dp)*0.031_dp)/m
    end do
    dx = 0.1_dp*x; dz = -0.2_dp*z; dw = -0.1_dp*weights
    available = omp_get_max_threads()
    call omp_set_num_threads(1)
    call rbf_apply(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], weights, reference, status)
    if (.not. status_ok(status)) error stop status%msg
    call rbf_apply_jvp(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], weights, &
        dx, dz, 0.2_dp, [0.1_dp, -0.1_dp, 0.2_dp], dw, tangent_ref, status)
    if (.not. status_ok(status)) error stop status%msg
    print '(a)', 'operation,threads,n,m,seconds,max_abs_error'
    threads = 1
    do
        call omp_set_num_threads(threads)
        elapsed = huge(1.0_dp)
        do repeat = 1, 3
            started = omp_get_wtime()
            call rbf_apply(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], &
                weights, result, status)
            elapsed = min(elapsed, omp_get_wtime() - started)
            if (.not. status_ok(status)) error stop status%msg
        end do
        call rbf_apply_jvp(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], weights, &
            dx, dz, 0.2_dp, [0.1_dp, -0.1_dp, 0.2_dp], dw, tangent, status)
        if (.not. status_ok(status)) error stop status%msg
        error = maxval(abs(result - reference))
        derror = maxval(abs(tangent - tangent_ref))
        print '(a,i0,2(a,i0),2(a,es17.9))', 'apply,', threads, ',', n, ',', m, &
            ',', elapsed, ',', error
        elapsed = huge(1.0_dp)
        do repeat = 1, 3
            started = omp_get_wtime()
            call rbf_apply_jvp(x, z, 1.4_dp, [0.7_dp, 1.2_dp, 0.9_dp], weights, &
                dx, dz, 0.2_dp, [0.1_dp, -0.1_dp, 0.2_dp], dw, tangent, status)
            elapsed = min(elapsed, omp_get_wtime() - started)
            if (.not. status_ok(status)) error stop status%msg
        end do
        print '(a,i0,2(a,i0),2(a,es17.9))', 'jvp,', threads, ',', n, ',', m, &
            ',', elapsed, ',', derror
        if (threads >= available) exit
        threads = min(2*threads, available)
    end do
end program
