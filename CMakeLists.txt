cmake_minimum_required(VERSION 3.15)

project(libcipr VERSION 1
        DESCRIPTION "Image Processing Library"
        LANGUAGES C
)

# Build options
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

include(FetchContent)
include(ExternalProject)

# Use modern timestamp behavior for FetchContent & ExternalProject
if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
endif()

# Get libspng with FetchContent
FetchContent_Declare(
        libspng 
        URL "https://github.com/randy408/libspng/archive/refs/tags/v0.7.4.tar.gz" #v0.7.4
)
FetchContent_MakeAvailable(libspng)

# Get libjpeg-turbo with ExternalProject
set(LIBJPEG_TURBO_PREFIX ${CMAKE_BINARY_DIR}/libjpeg-turbo)
set(LIBJPEG_TURBO_INSTALL_DIR ${LIBJPEG_TURBO_PREFIX}/install)
ExternalProject_add(libjpeg-turbo
        URL "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.2/libjpeg-turbo-3.1.2.tar.gz" # v3.1.2
        PREFIX ${LIBJPEG_TURBO_PREFIX}
        INSTALL_DIR ${LIBJPEG_TURBO_INSTALL_DIR}
        BUILD_BYPRODUCTS ${LIBJPEG_TURBO_INSTALL_DIR}/lib/libturbojpeg.a
        CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DENABLE_SHARED=OFF
                -DENABLE_STATIC=ON
)

# Make libjpeg-turbo install directories available at configure time
file (MAKE_DIRECTORY "${LIBJPEG_TURBO_INSTALL_DIR}/include")
file (MAKE_DIRECTORY "${LIBJPEG_TURBO_INSTALL_DIR}/lib")

# Create an import target for libjpeg-turbo
add_library(libjpeg-turbo_static STATIC IMPORTED GLOBAL)
set_target_properties(libjpeg-turbo_static PROPERTIES
        IMPORTED_LOCATION "${LIBJPEG_TURBO_INSTALL_DIR}/lib/libturbojpeg.a"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBJPEG_TURBO_INSTALL_DIR}/include"
)
add_dependencies(libjpeg-turbo_static libjpeg-turbo)

# libcipr sources
set(cipr_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/utils.c        
        ${CMAKE_CURRENT_SOURCE_DIR}/src/threading/task_queue.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/threading/thread_pool.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/image/image.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/io/layout_convert.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/io/png_io.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/io/jpeg_io.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/format/format_convert.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/filter/blur_gaussian.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/filter/blur_box.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/filter/unsharp_mask.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/filter/sobel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/filter/legacy/legacy_blur_gaussian.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/filter/legacy/legacy_blur_box.c
)

# Create the cipr static library target
add_library(cipr STATIC ${cipr_SOURCES})

target_include_directories(cipr
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set C standard
set_target_properties(cipr
        PROPERTIES
                CMAKE_C_STANDARD 11
                CMAKE_C_STANDARD_REQUIRED ON
)

# SIMD and optimization flags 
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_compile_options(cipr PRIVATE -O3 -march=native -mavx2 -mfma)
else()
        target_compile_options(cipr PRIVATE -O3 -march=native -mavx2 -mfma -flto)
endif()

# libjpeg-turbo must build first
add_dependencies(cipr libjpeg-turbo)

# Pthreads
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Math library
if(NOT WIN32)
        set(MATH_LIBRARY "m")
else()
        set(MATH_LIBRARY "")
endif()

# Link cipr with the dependencies, Pthreads and (optionally) the math library
target_link_libraries(cipr
        PRIVATE
                libjpeg-turbo_static 
                spng_static
                Threads::Threads
                ${MATH_LIBRARY}
)

# # Build the examples
if(BUILD_EXAMPLES)
        add_executable(eg_gaussian ${CMAKE_CURRENT_SOURCE_DIR}/examples/eg_gaussian.c)
        target_link_libraries(eg_gaussian cipr)

        add_executable(eg_box ${CMAKE_CURRENT_SOURCE_DIR}/examples/eg_box.c)
        target_link_libraries(eg_box cipr)
        
        add_executable(eg_unsharp_mask ${CMAKE_CURRENT_SOURCE_DIR}/examples/eg_unsharp_mask.c)
        target_link_libraries(eg_unsharp_mask cipr)

        add_executable(eg_sobel ${CMAKE_CURRENT_SOURCE_DIR}/examples/eg_sobel.c)
        target_link_libraries(eg_sobel cipr)

        add_executable(eg_combination ${CMAKE_CURRENT_SOURCE_DIR}/examples/eg_combination.c)
        target_link_libraries(eg_combination cipr)
endif()

# Build the benchmarks
if(BUILD_BENCHMARKS)
        add_executable(benchmark_gaussian ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark_gaussian.c)
        target_link_libraries(benchmark_gaussian cipr)

        add_executable(benchmark_box ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark_box.c)
        target_link_libraries(benchmark_box cipr)
endif()