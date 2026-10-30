module fortgp_basis
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_OK
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortgp, only: rbf_covariance, rbf_covariance_jvp
    implicit none
    private

    type, public :: rbf_basis_t
        real(dp), allocatable :: means(:), scales(:), centres(:, :)
        logical, allocatable :: active(:)
        real(dp) :: kernel_jitter = 1.0e-10_dp
        real(dp) :: length_multiplier = 1.0_dp
        real(dp), allocatable, private :: lengths(:)
        type(cholesky_factorization_t), private :: factor
        logical, private :: fitted = .false.
    contains
        procedure :: fit => basis_fit
        procedure :: evaluate => basis_evaluate
        procedure :: evaluate_jvp => basis_jvp
        procedure :: clear => basis_clear
    end type
contains
    subroutine invalid(status, message)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: message
        call status_set(status, FORTNUM_DOMAIN_ERROR, 'fortgp_basis: '//message)
    end subroutine

    subroutine basis_clear(self)
        class(rbf_basis_t), intent(inout) :: self
        self%fitted = .false.
        if (allocated(self%means)) deallocate(self%means)
        if (allocated(self%scales)) deallocate(self%scales)
        if (allocated(self%centres)) deallocate(self%centres)
        if (allocated(self%active)) deallocate(self%active)
        if (allocated(self%lengths)) deallocate(self%lengths)
        if (allocated(self%factor%lower)) deallocate(self%factor%lower)
    end subroutine

    subroutine basis_fit(self, x, ncentres, length_multiplier, status)
        class(rbf_basis_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), length_multiplier
        integer, intent(in) :: ncentres
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: z(:, :), selected(:, :), nearest(:), kernel(:, :)
        real(dp) :: distance, spread
        integer :: n, d, i, j, index, rank, limit, nactive
        call self%clear()
        call invalid(status, 'nonempty finite fitting inputs and positive rank required')
        n = size(x, 2)
        if (size(x, 1) < 1 .or. n < 1 .or. ncentres < 1) return
        if (.not. all(ieee_is_finite(x))) return
        call invalid(status, 'length multiplier and numerical jitter must be positive finite')
        if (.not. ieee_is_finite(length_multiplier)) return
        if (length_multiplier <= 0.0_dp) return
        if (.not. ieee_is_finite(self%kernel_jitter)) return
        if (self%kernel_jitter <= 0.0_dp) return
        call invalid(status, 'workspace would exceed four GiB')
        if (8_int64*(int(size(x, 1), int64)*n + &
            int(min(n, ncentres), int64)**2 + &
            int(size(x, 1), int64)*min(n, ncentres) + n) > &
            4_int64*1024**3) return
        self%length_multiplier = length_multiplier
        allocate(self%means(size(x, 1)), self%scales(size(x, 1)))
        allocate(self%active(size(x, 1)), z(size(x, 1), n))
        do d = 1, size(x, 1)
            ! Scale before summation to avoid overflowing the mean.
            self%means(d) = sum(x(d, :)/real(n, dp))
            z(d, :) = x(d, :) - self%means(d)
            spread = maxval(abs(z(d, :)))
            self%active(d) = maxval(x(d, :)) > minval(x(d, :))
            self%scales(d) = 1.0_dp
            if (self%active(d)) then
                self%scales(d) = spread*sqrt(sum((z(d, :)/spread)**2)/real(n, dp))
                z(d, :) = z(d, :)/self%scales(d)
            else
                z(d, :) = 0.0_dp
            end if
        end do
        call invalid(status, 'standardization overflow')
        if (.not. all(ieee_is_finite(z))) return
        if (.not. all(ieee_is_finite(self%scales))) return
        nactive = max(1, count(self%active))
        allocate(self%lengths(size(x, 1)), &
            source=sqrt(real(nactive, dp))*length_multiplier)
        limit = min(n, ncentres)
        allocate(selected(size(x, 1), limit), nearest(n))
        do i = 1, n
            nearest(i) = sum(z(:, i)**2)
        end do
        index = minloc(nearest, dim=1)
        nearest = huge(1.0_dp)
        rank = 0
        do j = 1, limit
            rank = j
            selected(:, j) = z(:, index)
            !$omp parallel do private(distance)
            do i = 1, n
                distance = sum((z(:, i) - selected(:, j))**2)
                nearest(i) = min(nearest(i), distance)
            end do
            !$omp end parallel do
            index = maxloc(nearest, dim=1)
            if (nearest(index) <= 0.0_dp) exit
        end do
        self%centres = selected(:, :rank)
        call rbf_covariance(self%centres, self%centres, 1.0_dp, &
            self%lengths, kernel, status)
        if (.not. status_ok(status)) return
        do i = 1, rank
            kernel(i, i) = kernel(i, i) + self%kernel_jitter
        end do
        call self%factor%factorize(kernel, status)
        if (.not. status_ok(status)) return
        self%fitted = .true.
    end subroutine

    logical function standardize(self, x, z, status) result(valid)
        class(rbf_basis_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: z(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: d
        valid = .false.
        call invalid(status, 'fit must succeed before evaluation')
        if (.not. self%fitted) return
        call invalid(status, 'query shape or finite values invalid')
        if (size(x, 1) /= size(self%means) .or. size(x, 2) < 1) return
        if (.not. all(ieee_is_finite(x))) return
        call invalid(status, 'query workspace would exceed four GiB')
        if (8_int64*int(size(x, 2), int64)*(4_int64*size(self%centres, 2) + &
            2_int64*size(x, 1)) > 4_int64*1024**3) return
        allocate(z(size(x, 1), size(x, 2)), source=0.0_dp)
        do d = 1, size(x, 1)
            if (self%active(d)) z(d, :) = (x(d, :) - self%means(d))/self%scales(d)
        end do
        if (.not. all(ieee_is_finite(z))) return
        call status_set(status, FORTNUM_OK, '')
        valid = .true.
    end function

    subroutine whiten(self, kernel, phi, status)
        class(rbf_basis_t), intent(in) :: self
        real(dp), intent(in) :: kernel(:, :)
        real(dp), allocatable, intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: work(:, :)
        work = transpose(kernel)
        call self%factor%solve_lower_matrix(work, status)
        if (status_ok(status)) phi = transpose(work)
    end subroutine

    subroutine basis_evaluate(self, x, phi, status)
        class(rbf_basis_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: z(:, :), kernel(:, :)
        if (.not. standardize(self, x, z, status)) return
        call rbf_covariance(z, self%centres, 1.0_dp, self%lengths, kernel, status)
        if (.not. status_ok(status)) return
        call whiten(self, kernel, phi, status)
    end subroutine

    subroutine basis_jvp(self, x, dx, dphi, status)
        class(rbf_basis_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), dx(:, :)
        real(dp), allocatable, intent(out) :: dphi(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: z(:, :), dz(:, :), dc(:, :), dl(:), kernel(:, :)
        integer :: d
        if (.not. standardize(self, x, z, status)) return
        call invalid(status, 'query tangent shape or finite values invalid')
        if (any(shape(dx) /= shape(x))) return
        if (.not. all(ieee_is_finite(dx))) return
        allocate(dz(size(x, 1), size(x, 2)), source=0.0_dp)
        do d = 1, size(x, 1)
            if (self%active(d)) dz(d, :) = dx(d, :)/self%scales(d)
        end do
        allocate(dc(size(self%centres, 1), size(self%centres, 2)), source=0.0_dp)
        allocate(dl(size(self%lengths)), source=0.0_dp)
        call rbf_covariance_jvp(z, self%centres, 1.0_dp, self%lengths, &
            dz, dc, 0.0_dp, dl, kernel, status)
        if (.not. status_ok(status)) return
        call whiten(self, kernel, dphi, status)
    end subroutine
end module fortgp_basis
