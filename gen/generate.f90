program generate
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortsym_arena, only: arena_t
    use fortsym_expr, only: expr_t, sym, exp_expr => exp, operator(*), &
        operator(-), operator(/), operator(**)
    use fortsym_diff, only: diff
    use fortsym_engine, only: engine_result_t
    use fortsym_engine_native, only: make_native_engine, native_engine_t
    use fortsym_kernel_emit, only: kernel_emit_spec_t, emit_fortran_kernel_ir
    use fortsym_kernel_ir, only: kernel_ir_t, lower_kernel_ir
    use fortsym_string, only: chars, str, str_t
    implicit none
    type(arena_t), target :: arena
    type(native_engine_t) :: engine
    type(expr_t) :: a, q, x, y, ell, value
    type(expr_t), allocatable :: roots(:)
    type(engine_result_t) :: reduced
    type(kernel_ir_t) :: ir
    type(kernel_emit_spec_t) :: spec
    type(str_t) :: source
    character(:), allocatable :: message
    integer :: i, unit, leaf
    logical :: ok

    call arena%init()
    engine = make_native_engine(arena)
    do leaf = 1, 2
        if (leaf == 1) then
            a = sym(arena, 'amplitude')
            q = sym(arena, 'distance')
            value = a**2*exp_expr(-0.5_dp*q)
            roots = [value, diff(value, a), diff(value, q)]
            spec%name = str('fortgp_rbf_leaf')
            spec%args = [str('amplitude'), str('distance')]
            spec%outputs = [str('value'), str('damplitude'), str('ddistance')]
        else
            x = sym(arena, 'x')
            y = sym(arena, 'y')
            ell = sym(arena, 'length')
            value = ((x - y)/ell)**2
            roots = [value, diff(value, x), diff(value, y), diff(value, ell)]
            spec%name = str('fortgp_distance_leaf')
            spec%args = [str('x'), str('y'), str('length')]
            spec%outputs = [str('value'), str('dx'), str('dy'), str('dlength')]
        end if
        do i = 1, size(roots)
            reduced = engine%simplify(roots(i))
            if (.not. reduced%ok) error stop 'FortSym simplification failed'
            roots(i) = reduced%value
        end do
        call lower_kernel_ir(roots, ir, ok, message)
        if (.not. ok) error stop message
        spec%generator = str('fortgp/gen/generate.f90')
        spec%generator_revision = str('fortsym 45ea8959002eedc8a85bb2dc386af293bb66a761')
        spec%regenerate_command = str('cd gen && fpm run generate')
        spec%temp_prefix = str('t')
        spec%pure_procedure = .true.
        source = emit_fortran_kernel_ir(ir, spec, ok, message)
        if (.not. ok) error stop message
        open (newunit=unit, file='../src/generated/'//chars(spec%name)//'.f90', &
            status='replace', action='write')
        write (unit, '(a)', advance='no') chars(source)
        close (unit)
    end do
end program generate
