cmake_minimum_required(VERSION 3.22)

project(fortgp LANGUAGES Fortran)

option(FORTGP_BUILD_TESTING "Build the fortgp tests" ${PROJECT_IS_TOP_LEVEL})

include(FetchContent)
# A consumer that already has fortinfer or fortnum targets (TIAGO) keeps them.
set(FORTGP_FORTINFER_TAG "abe55209bf699bb0d1b3d874dc9adf3c165e018b"
    CACHE STRING "fortinfer commit used by fortgp")
if(NOT TARGET fortinfer)
    FetchContent_Declare(fortinfer
        GIT_REPOSITORY https://github.com/lazy-fortran/fortinfer.git
        GIT_TAG ${FORTGP_FORTINFER_TAG})
    FetchContent_MakeAvailable(fortinfer)
endif()
# OpenMP parallelizes covariance assembly when the compiler supports it.
find_package(OpenMP COMPONENTS Fortran)

add_library(fortgp STATIC
    src/generated/fortgp_autoregressive_leaf.f90
    src/generated/fortgp_distance_leaf.f90
    src/generated/fortgp_matern32_leaf.f90
    src/generated/fortgp_matern52_leaf.f90
    src/generated/fortgp_rbf_leaf.f90
    src/fortgp.f90
    src/fortgp_ar1.f90
    src/fortgp_basis.f90
    src/fortgp_gaussian_constraints.f90
    src/fortgp_multifidelity.f90
    src/fortgp_reductions.f90)
set_target_properties(fortgp PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod)
target_include_directories(fortgp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>)
target_link_libraries(fortgp PUBLIC fortinfer fortnum)
if(OpenMP_Fortran_FOUND)
    target_link_libraries(fortgp PUBLIC OpenMP::OpenMP_Fortran)
endif()

if(FORTGP_BUILD_TESTING)
    include(CTest)
    if(NOT TARGET test-drive)
        FetchContent_Declare(test-drive
            GIT_REPOSITORY https://github.com/fortran-lang/test-drive.git
            GIT_TAG 791484a79a24cbc00be6f74e53c6e030cb75efe2)
        set(TESTDRIVE_BUILD_TESTING OFF)
        FetchContent_MakeAvailable(test-drive)
    endif()
    foreach(name basis exact gaussian_constraints matern multifidelity reductions)
        add_executable(test_${name} test/test_${name}.f90)
        set_target_properties(test_${name} PROPERTIES
            Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod/test_${name})
        target_link_libraries(test_${name} PRIVATE fortgp test-drive)
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
endif()
