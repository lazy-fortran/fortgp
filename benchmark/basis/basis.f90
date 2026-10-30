program benchmark_basis
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use omp_lib, only: omp_get_wtime, omp_set_num_threads
    use fortgp_basis, only: rbf_basis_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none
    type(rbf_basis_t) :: basis
    type(fortnum_status_t) :: status
    integer, parameter :: n = 100000, ndim = 8, rank = 64
    real(dp), allocatable :: x(:, :), phi(:, :), reference(:, :)
    real(dp) :: start, fit_time, evaluate_time, difference
    integer :: i, d, threads
    allocate(x(ndim,n))
    do i = 1, n
        do d = 1, ndim
            x(d,i) = sin(real(i*d,dp)*0.001_dp)+cos(real(i*(d+1),dp)*0.003_dp)
        end do
    end do
    print '(a)', 'threads,fit_seconds,evaluate_seconds,max_abs_difference'
    do threads = 1, 4, 3
        call omp_set_num_threads(threads)
        start = omp_get_wtime()
        call basis%fit(x, rank, 1.0_dp, status)
        if (.not. status_ok(status)) error stop status%msg
        fit_time = omp_get_wtime()-start
        start = omp_get_wtime()
        call basis%evaluate(x, phi, status)
        if (.not. status_ok(status)) error stop status%msg
        evaluate_time = omp_get_wtime()-start
        difference = 0.0_dp
        if (threads == 1) reference = phi
        if (threads /= 1) difference = maxval(abs(phi-reference))
        print '(i0,3(a,es16.8))', threads, ',', fit_time, ',', evaluate_time, ',', difference
    end do
end program
