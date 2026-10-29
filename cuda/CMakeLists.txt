cmake_minimum_required(VERSION 3.24)
project(fortgp_cuda_backend LANGUAGES CXX CUDA)
add_library(fortgp_cuda_backend STATIC backend/reductions.cu)
set_target_properties(fortgp_cuda_backend PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE YES)
install(TARGETS fortgp_cuda_backend ARCHIVE DESTINATION lib)
