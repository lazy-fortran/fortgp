module fortgp
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_OK
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortprob_gaussian, only: gaussian_conditioner_t, gaussian_condition_jvp, &
        gaussian_logpdf, validate_covariance
    implicit none
    private

    type, public :: exact_gp_t
        private
        real(dp), allocatable :: x(:, :), y(:), length(:), covariance(:, :), zero(:)
        real(dp) :: amplitude = 0.0_dp
        type(gaussian_conditioner_t), allocatable :: conditioner
        logical :: fitted = .false.
    contains
        procedure :: fit
        procedure :: predict
        procedure :: predict_jvp
        procedure :: predict_y_vjp
        procedure :: log_marginal_likelihood
        procedure :: log_marginal_likelihood_jvp
    end type exact_gp_t
    public :: rbf_covariance, rbf_covariance_jvp
    public :: matern32_covariance, matern32_covariance_jvp
    public :: matern52_covariance, matern52_covariance_jvp

    interface
        pure subroutine fortgp_rbf_leaf(amplitude, distance, value, &
                damplitude, ddistance)
            import dp
            real(dp), intent(in) :: amplitude, distance
            real(dp), intent(out) :: value, damplitude, ddistance
        end subroutine fortgp_rbf_leaf
        pure subroutine fortgp_matern32_leaf(amplitude, distance, value, &
                damplitude, ddistance)
            import dp
            real(dp), intent(in) :: amplitude, distance
            real(dp), intent(out) :: value, damplitude, ddistance
        end subroutine fortgp_matern32_leaf
        pure subroutine fortgp_matern52_leaf(amplitude, distance, value, &
                damplitude, ddistance)
            import dp
            real(dp), intent(in) :: amplitude, distance
            real(dp), intent(out) :: value, damplitude, ddistance
        end subroutine fortgp_matern52_leaf
        pure subroutine fortgp_distance_leaf(x, y, length, value, dx, dy, dlength)
            import dp
            real(dp), intent(in) :: x, y, length
            real(dp), intent(out) :: value, dx, dy, dlength
        end subroutine fortgp_distance_leaf
    end interface

contains

    subroutine invalid(status, message)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: message
        call status_set(status, FORTNUM_DOMAIN_ERROR, 'fortgp: '//message)
    end subroutine invalid

    logical function valid_inputs(x, amplitude, length, status) result(valid)
        real(dp), intent(in) :: x(:, :), amplitude, length(:)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        call invalid(status, 'nonempty x(ndim,npoints) and length(ndim) required')
        if (size(x, 1) < 1 .or. size(x, 2) < 1) return
        if (size(length) /= size(x, 1)) return
        call invalid(status, 'finite inputs, positive amplitude and lengths required')
        if (.not. all(ieee_is_finite(x))) return
        if (.not. all(ieee_is_finite(length))) return
        if (.not. ieee_is_finite(amplitude)) return
        if (amplitude <= 0.0_dp .or. any(length <= 0.0_dp)) return
        valid = .true.
        call status_set(status, FORTNUM_OK, '')
    end function valid_inputs

    subroutine rbf_covariance(x, z, amplitude, length, covariance, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:)
        real(dp), allocatable, intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, term, dx, dy, dl, da, dq
        integer :: i, j, d
        if (.not. valid_inputs(x, amplitude, length, status)) return
        if (.not. valid_inputs(z, amplitude, length, status)) return
        allocate (covariance(size(x, 2), size(z, 2)))
        !$omp parallel do collapse(2) private(distance,term,dx,dy,dl,da,dq,d)
        do j = 1, size(z, 2)
            do i = 1, size(x, 2)
                distance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, dx, dy, dl)
                    distance = distance + term
                end do
                call fortgp_rbf_leaf(amplitude, distance, covariance(i, j), da, dq)
            end do
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(covariance))) then
            deallocate (covariance)
            call invalid(status, 'kernel evaluation overflow or invalid arithmetic')
        end if
    end subroutine rbf_covariance

    subroutine rbf_covariance_jvp(x, z, amplitude, length, dx, dz, &
            damplitude, dlength, dcovariance, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:)
        real(dp), intent(in) :: dx(:, :), dz(:, :), damplitude, dlength(:)
        real(dp), allocatable, intent(out) :: dcovariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, ddistance, term, gx, gz, gl, ga, gq, value
        integer :: i, j, d
        if (.not. valid_inputs(x, amplitude, length, status)) return
        if (.not. valid_inputs(z, amplitude, length, status)) return
        call invalid(status, 'kernel tangent shapes differ from primal shapes')
        if (any(shape(dx) /= shape(x)) .or. any(shape(dz) /= shape(z))) return
        if (size(dlength) /= size(length)) return
        call invalid(status, 'kernel tangents must be finite')
        if (.not. all(ieee_is_finite(dx))) return
        if (.not. all(ieee_is_finite(dz))) return
        if (.not. all(ieee_is_finite(dlength))) return
        if (.not. ieee_is_finite(damplitude)) return
        allocate (dcovariance(size(x, 2), size(z, 2)))
        !$omp parallel do collapse(2) private(distance,ddistance,term,gx,gz,gl,ga,gq,value,d)
        do j = 1, size(z, 2)
            do i = 1, size(x, 2)
                distance = 0.0_dp
                ddistance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, gx, gz, gl)
                    distance = distance + term
                    ddistance = ddistance + gx*dx(d, i) + gz*dz(d, j) + gl*dlength(d)
                end do
                call fortgp_rbf_leaf(amplitude, distance, value, ga, gq)
                dcovariance(i, j) = ga*damplitude + gq*ddistance
            end do
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(dcovariance))) then
            deallocate (dcovariance)
            call invalid(status, 'kernel tangent evaluation overflow')
            return
        end if
        call status_set(status, FORTNUM_OK, '')
    end subroutine rbf_covariance_jvp

    subroutine matern32_covariance(x, z, amplitude, length, covariance, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:)
        real(dp), allocatable, intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, term, dx, dy, dl, da, dq
        integer :: i, j, d
        if (.not. valid_inputs(x, amplitude, length, status)) return
        if (.not. valid_inputs(z, amplitude, length, status)) return
        allocate (covariance(size(x, 2), size(z, 2)))
        !$omp parallel do collapse(2) private(distance,term,dx,dy,dl,da,dq,d)
        do j = 1, size(z, 2)
            do i = 1, size(x, 2)
                distance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, dx, dy, dl)
                    distance = distance + term
                end do
                call fortgp_matern32_leaf(amplitude, sqrt(distance), covariance(i, j), da, dq)
            end do
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(covariance))) then
            deallocate (covariance)
            call invalid(status, 'kernel evaluation overflow or invalid arithmetic')
        end if
    end subroutine matern32_covariance

    subroutine matern32_covariance_jvp(x, z, amplitude, length, dx, dz, &
            damplitude, dlength, dcovariance, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:)
        real(dp), intent(in) :: dx(:, :), dz(:, :), damplitude, dlength(:)
        real(dp), allocatable, intent(out) :: dcovariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, ddistance, term, gx, gz, gl, ga, gq, value
        integer :: i, j, d
        if (.not. valid_inputs(x, amplitude, length, status)) return
        if (.not. valid_inputs(z, amplitude, length, status)) return
        call invalid(status, 'kernel tangent shapes differ from primal shapes')
        if (any(shape(dx) /= shape(x)) .or. any(shape(dz) /= shape(z))) return
        if (size(dlength) /= size(length)) return
        call invalid(status, 'kernel tangents must be finite')
        if (.not. all(ieee_is_finite(dx))) return
        if (.not. all(ieee_is_finite(dz))) return
        if (.not. all(ieee_is_finite(dlength))) return
        if (.not. ieee_is_finite(damplitude)) return
        allocate (dcovariance(size(x, 2), size(z, 2)))
        !$omp parallel do collapse(2) private(distance,ddistance,term,gx,gz,gl,ga,gq,value,d)
        do j = 1, size(z, 2)
            do i = 1, size(x, 2)
                distance = 0.0_dp
                ddistance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, gx, gz, gl)
                    distance = distance + term
                    ddistance = ddistance + gx*dx(d, i) + gz*dz(d, j) + gl*dlength(d)
                end do
                call fortgp_matern32_leaf(amplitude, sqrt(distance), value, ga, gq)
                dcovariance(i, j) = ga*damplitude + gq*ddistance
            end do
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(dcovariance))) then
            deallocate (dcovariance)
            call invalid(status, 'kernel tangent evaluation overflow')
            return
        end if
        call status_set(status, FORTNUM_OK, '')
    end subroutine matern32_covariance_jvp

    subroutine matern52_covariance(x, z, amplitude, length, covariance, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:)
        real(dp), allocatable, intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, term, dx, dy, dl, da, dq
        integer :: i, j, d
        if (.not. valid_inputs(x, amplitude, length, status)) return
        if (.not. valid_inputs(z, amplitude, length, status)) return
        allocate (covariance(size(x, 2), size(z, 2)))
        !$omp parallel do collapse(2) private(distance,term,dx,dy,dl,da,dq,d)
        do j = 1, size(z, 2)
            do i = 1, size(x, 2)
                distance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, dx, dy, dl)
                    distance = distance + term
                end do
                call fortgp_matern52_leaf(amplitude, sqrt(distance), covariance(i, j), da, dq)
            end do
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(covariance))) then
            deallocate (covariance)
            call invalid(status, 'kernel evaluation overflow or invalid arithmetic')
        end if
    end subroutine matern52_covariance

    subroutine matern52_covariance_jvp(x, z, amplitude, length, dx, dz, &
            damplitude, dlength, dcovariance, status)
        real(dp), intent(in) :: x(:, :), z(:, :), amplitude, length(:)
        real(dp), intent(in) :: dx(:, :), dz(:, :), damplitude, dlength(:)
        real(dp), allocatable, intent(out) :: dcovariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: distance, ddistance, term, gx, gz, gl, ga, gq, value
        integer :: i, j, d
        if (.not. valid_inputs(x, amplitude, length, status)) return
        if (.not. valid_inputs(z, amplitude, length, status)) return
        call invalid(status, 'kernel tangent shapes differ from primal shapes')
        if (any(shape(dx) /= shape(x)) .or. any(shape(dz) /= shape(z))) return
        if (size(dlength) /= size(length)) return
        call invalid(status, 'kernel tangents must be finite')
        if (.not. all(ieee_is_finite(dx))) return
        if (.not. all(ieee_is_finite(dz))) return
        if (.not. all(ieee_is_finite(dlength))) return
        if (.not. ieee_is_finite(damplitude)) return
        allocate (dcovariance(size(x, 2), size(z, 2)))
        !$omp parallel do collapse(2) private(distance,ddistance,term,gx,gz,gl,ga,gq,value,d)
        do j = 1, size(z, 2)
            do i = 1, size(x, 2)
                distance = 0.0_dp
                ddistance = 0.0_dp
                do d = 1, size(x, 1)
                    call fortgp_distance_leaf(x(d, i), z(d, j), length(d), &
                        term, gx, gz, gl)
                    distance = distance + term
                    ddistance = ddistance + gx*dx(d, i) + gz*dz(d, j) + gl*dlength(d)
                end do
                call fortgp_matern52_leaf(amplitude, sqrt(distance), value, ga, gq)
                dcovariance(i, j) = ga*damplitude + gq*ddistance
            end do
        end do
        !$omp end parallel do
        if (.not. all(ieee_is_finite(dcovariance))) then
            deallocate (dcovariance)
            call invalid(status, 'kernel tangent evaluation overflow')
            return
        end if
        call status_set(status, FORTNUM_OK, '')
    end subroutine matern52_covariance_jvp

    subroutine fit(self, x, y, noise_cov, amplitude, length, status)
        class(exact_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:), noise_cov(:, :), amplitude, length(:)
        type(fortnum_status_t), intent(out) :: status
        type(gaussian_conditioner_t) :: conditioner
        real(dp), allocatable :: covariance(:, :), zero(:)
        integer :: n
        self%fitted = .false.
        if (allocated(self%conditioner)) deallocate(self%conditioner)
        if (.not. valid_inputs(x, amplitude, length, status)) return
        n = size(x, 2)
        call invalid(status, 'y or noise covariance shape differs from training data')
        if (size(y) /= n) return
        if (size(noise_cov, 1) /= n .or. size(noise_cov, 2) /= n) return
        call invalid(status, 'observations must be finite')
        if (.not. all(ieee_is_finite(y))) return
        call validate_covariance(noise_cov, status)
        if (.not. status_ok(status)) return
        call rbf_covariance(x, x, amplitude, length, covariance, status)
        if (.not. status_ok(status)) return
        covariance = covariance + noise_cov
        allocate(zero(n), source=0.0_dp)
        call conditioner%fit(zero, covariance, y, status)
        if (.not. status_ok(status)) return
        self%x = x
        self%y = y
        call move_alloc(zero, self%zero)
        self%length = length
        self%amplitude = amplitude
        call move_alloc(covariance, self%covariance)
        allocate(self%conditioner, source=conditioner)
        self%fitted = .true.
    end subroutine fit

    logical function valid_query(self, query, status) result(valid)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        call invalid(status, 'fit must succeed before prediction')
        if (.not. self%fitted) return
        valid = valid_inputs(query, self%amplitude, self%length, status)
    end function valid_query

    subroutine predict(self, query, mean, latent_cov, status, &
            future_noise_cov, observation_cov)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :)
        real(dp), allocatable, intent(out) :: mean(:), latent_cov(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), optional, intent(in) :: future_noise_cov(:, :)
        real(dp), allocatable, optional, intent(out) :: observation_cov(:, :)
        real(dp), allocatable :: prior(:, :), cross(:, :), zero(:)
        integer :: m
        if (.not. valid_query(self, query, status)) return
        m = size(query, 2)
        if (present(future_noise_cov)) then
            call invalid(status, 'future noise covariance shape differs from query')
            if (size(future_noise_cov, 1) /= m .or. size(future_noise_cov, 2) /= m) return
            call validate_covariance(future_noise_cov, status)
            if (.not. status_ok(status)) return
        end if
        call rbf_covariance(query, query, self%amplitude, self%length, prior, status)
        if (.not. status_ok(status)) return
        call rbf_covariance(query, self%x, self%amplitude, self%length, cross, status)
        if (.not. status_ok(status)) return
        allocate(zero(m), source=0.0_dp)
        call self%conditioner%condition(zero, prior, cross, mean, latent_cov, status)
        if (.not. status_ok(status)) return
        if (present(observation_cov)) then
            observation_cov = latent_cov
            if (present(future_noise_cov)) &
                observation_cov = observation_cov + future_noise_cov
            if (.not. all(ieee_is_finite(observation_cov))) then
                deallocate (mean, latent_cov, observation_cov)
                call invalid(status, 'observation covariance overflow')
            end if
        end if
    end subroutine predict

    subroutine training_tangent(self, dx, dy, dnoise, damplitude, dlength, &
            dcov, status)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(in) :: dx(:, :), dy(:), dnoise(:, :), damplitude, dlength(:)
        real(dp), allocatable, intent(out) :: dcov(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n
        call invalid(status, 'fit must succeed before differentiation')
        if (.not. self%fitted) return
        n = size(self%y)
        call invalid(status, 'observation or noise tangent shape differs from training')
        if (size(dy) /= n) return
        if (size(dnoise, 1) /= n .or. size(dnoise, 2) /= n) return
        call invalid(status, 'observation and noise tangents must be finite and symmetric')
        if (.not. all(ieee_is_finite(dy))) return
        if (.not. all(ieee_is_finite(dnoise))) return
        if (maxval(abs(dnoise - transpose(dnoise))) > &
            128*epsilon(1.0_dp)*n*maxval(abs(dnoise))) return
        call rbf_covariance_jvp(self%x, self%x, self%amplitude, self%length, &
            dx, dx, damplitude, dlength, dcov, status)
        if (.not. status_ok(status)) return
        dcov = dcov + dnoise
    end subroutine training_tangent

    subroutine predict_jvp(self, query, dquery, dx, dy, dnoise, damplitude, &
            dlength, dmean, dlatent_cov, status)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), dquery(:, :), dx(:, :), dy(:)
        real(dp), intent(in) :: dnoise(:, :), damplitude, dlength(:)
        real(dp), allocatable, intent(out) :: dmean(:), dlatent_cov(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: prior(:, :), cross(:, :), zero(:), dcov(:, :)
        real(dp), allocatable :: dprior(:, :), dcross(:, :)
        integer :: m
        if (.not. valid_query(self, query, status)) return
        m = size(query, 2)
        call training_tangent(self, dx, dy, dnoise, damplitude, dlength, dcov, status)
        if (.not. status_ok(status)) return
        call rbf_covariance(query, query, self%amplitude, self%length, prior, status)
        if (.not. status_ok(status)) return
        call rbf_covariance(query, self%x, self%amplitude, self%length, cross, status)
        if (.not. status_ok(status)) return
        call rbf_covariance_jvp(query, query, self%amplitude, self%length, &
            dquery, dquery, damplitude, dlength, dprior, status)
        if (.not. status_ok(status)) return
        call rbf_covariance_jvp(query, self%x, self%amplitude, self%length, &
            dquery, dx, damplitude, dlength, dcross, status)
        if (.not. status_ok(status)) return
        allocate(zero(m), source=0.0_dp)
        call gaussian_condition_jvp(zero, prior, &
            self%zero, self%covariance, cross, self%y, &
            zero, dprior, self%zero, dcov, &
            dcross, dy, dmean, dlatent_cov, status)
    end subroutine predict_jvp

    subroutine predict_y_vjp(self, query, mean_weight, y_gradient, status)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), mean_weight(:)
        real(dp), allocatable, intent(out) :: y_gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factor
        real(dp), allocatable :: cross(:, :)
        if (.not. valid_query(self, query, status)) return
        call invalid(status, 'mean weight must be finite and match query count')
        if (size(mean_weight) /= size(query, 2)) return
        if (.not. all(ieee_is_finite(mean_weight))) return
        call rbf_covariance(query, self%x, self%amplitude, self%length, cross, status)
        if (.not. status_ok(status)) return
        y_gradient = matmul(transpose(cross), mean_weight)
        if (.not. all(ieee_is_finite(y_gradient))) then
            deallocate(y_gradient)
            call invalid(status, 'observation transpose product overflow')
            return
        end if
        call factor%factorize(self%covariance, status)
        if (.not. status_ok(status)) return
        call factor%solve(y_gradient, status)
        if (.not. all(ieee_is_finite(y_gradient))) then
            deallocate(y_gradient)
            call invalid(status, 'observation transpose solve overflow')
        end if
    end subroutine predict_y_vjp

    subroutine log_marginal_likelihood(self, value, status)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        value = 0.0_dp
        call invalid(status, 'fit must succeed before likelihood evaluation')
        if (.not. self%fitted) return
        call gaussian_logpdf(self%y, self%zero, &
            self%covariance, value, status)
    end subroutine log_marginal_likelihood

    subroutine log_marginal_likelihood_jvp(self, dx, dy, dnoise, damplitude, &
            dlength, dvalue, status)
        class(exact_gp_t), intent(in) :: self
        real(dp), intent(in) :: dx(:, :), dy(:), dnoise(:, :), damplitude, dlength(:)
        real(dp), intent(out) :: dvalue
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factor
        real(dp), allocatable :: dcov(:, :), alpha(:), solved(:, :), product(:)
        integer :: i
        dvalue = 0.0_dp
        call training_tangent(self, dx, dy, dnoise, damplitude, dlength, dcov, status)
        if (.not. status_ok(status)) return
        call factor%factorize(self%covariance, status)
        if (.not. status_ok(status)) return
        alpha = self%y
        call factor%solve(alpha, status)
        if (.not. status_ok(status)) return
        solved = dcov
        call factor%solve(solved, status)
        if (.not. status_ok(status)) return
        product = matmul(dcov, alpha)
        dvalue = 0.5_dp*dot_product(alpha, product) - dot_product(dy, alpha)
        do i = 1, size(alpha)
            dvalue = dvalue - 0.5_dp*solved(i, i)
        end do
        if (.not. ieee_is_finite(dvalue)) &
            call invalid(status, 'likelihood tangent overflow')
    end subroutine log_marginal_likelihood_jvp

end module fortgp
