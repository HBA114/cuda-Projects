cmake_minimum_required(VERSION 3.22.1)

project(HelloWorld LANGUAGES CXX CUDA)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")
add_library(hello STATIC
    hello.cu
)

target_compile_features(hello PUBLIC cxx_std_23)
set_target_properties(hello PROPERTIES CUDA_ARCHITECTURES "50")

add_executable(hello_test hello.cu)
 
set_property(TARGET hello_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hello_test PRIVATE hello)