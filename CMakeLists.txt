cmake_minimum_required(VERSION 3.12)
project(qvcache LANGUAGES CXX)

set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler options first!
add_compile_options(
    -w
    -march=native
    -Wall
    -mavx
    -fno-builtin-malloc
    -fno-builtin-calloc
    -fno-builtin-realloc
    -fno-builtin-free
    -fopenmp
    -fopenmp-simd
    -funroll-loops
    -Wfatal-errors
    -DUSE_ACCELERATED_PQ
    -DUSE_AVX2
    -Ofast
    -march=native 
    -mtune=native
    -DNDEBUG
)

# Then find dependencies
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost COMPONENTS program_options)
find_package(spdlog REQUIRED)
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense PATHS /usr/include/eigen3 /usr/local/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Eigen3. Please install libeigen3-dev or set EIGEN3_INCLUDE_DIR.")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        # Get pybind11 CMake directory from Python (pybind11 installed via pip)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE PYTHON_RESULT
        )
        if(PYTHON_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
            if(EXISTS "${PYBIND11_CMAKE_DIR}")
                # Set pybind11_DIR directly to the CMake directory
                # This directory should contain pybind11Config.cmake
                set(pybind11_DIR "${PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake directory" FORCE)
                message(STATUS "Found pybind11 CMake dir via Python: ${PYBIND11_CMAKE_DIR}")
                message(STATUS "Set pybind11_DIR to: ${pybind11_DIR}")
            endif()
        endif()
    endif()
    
    # Find pybind11 package (should now work with CMAKE_PREFIX_PATH set)
    find_package(pybind11 QUIET)
    
    if(pybind11_FOUND)
        message(STATUS "pybind11 found, building Python bindings")
        add_subdirectory(python)
    else()
        message(STATUS "pybind11 not found, skipping Python bindings")
        message(STATUS "Install pybind11 with: pip install pybind11")
        message(STATUS "If installed, check: python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'")
    endif()
endif()

# Then add subdirectories
add_subdirectory(backends/Greator)
add_subdirectory(external/DiskANN)
add_subdirectory(benchmarks)
add_subdirectory(src)
add_subdirectory(utils)