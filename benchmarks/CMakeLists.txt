# List of test executable names and their source files
        set(TEST_EXECUTABLES
            build_index diskann/build_index.cpp
            qvcache_benchmark_diskann qvcache/qvcache_benchmark_diskann.cpp
            qvcache_benchmark_sptag qvcache/qvcache_benchmark_sptag.cpp
            backend_benchmark_diskann qvcache/backend_benchmark_diskann.cpp
            backend_benchmark_sptag qvcache/backend_benchmark_sptag.cpp
        )

# Compute length of the list
list(LENGTH TEST_EXECUTABLES len)
math(EXPR max_index "${len} - 1")

# Loop through the list, two items at a time
foreach(i RANGE 0 ${max_index} 2)
    list(GET TEST_EXECUTABLES ${i} exec_name)
    math(EXPR src_index "${i} + 1")
    list(GET TEST_EXECUTABLES ${src_index} src_file)

    add_executable(${exec_name} ${src_file})

    target_link_libraries(${exec_name}
        PRIVATE
            greator
            diskann
            qvcache
            TBB::tbb
            OpenMP::OpenMP_CXX
            Boost::program_options
            spdlog::spdlog
)

    target_include_directories(${exec_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/backends
            ${PROJECT_SOURCE_DIR}/backends/Greator/include
            ${PROJECT_SOURCE_DIR}/external/DiskANN/include
    )
endforeach()

# Aker: DiskANN benchmark that uses libaker.so as the result cache
set(AKER_ROOT "${PROJECT_SOURCE_DIR}/Aker")
find_library(AKER_LIBRARY
    NAMES aker
    PATHS "${AKER_ROOT}/build/lib" /usr/local/lib
)
if(AKER_LIBRARY AND EXISTS "${AKER_ROOT}/inc/ak_anns_cache_c_wrapper.h")
    add_executable(aker_benchmark_diskann aker/aker_benchmark_diskann.cpp)
    add_executable(aker_approx_hit_probe aker/aker_approx_hit_probe.cpp)
    foreach(_aker_exe aker_benchmark_diskann aker_approx_hit_probe)
        target_link_libraries(${_aker_exe}
            PRIVATE
                greator
                diskann
                ${AKER_LIBRARY}
                TBB::tbb
                OpenMP::OpenMP_CXX
                Boost::program_options
                spdlog::spdlog
        )
        target_include_directories(${_aker_exe}
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
                ${PROJECT_SOURCE_DIR}/backends
                ${PROJECT_SOURCE_DIR}/backends/Greator/include
                ${PROJECT_SOURCE_DIR}/external/DiskANN/include
                ${AKER_ROOT}/inc
                ${PROJECT_SOURCE_DIR}/benchmarks/aker
        )
        # Aker: libaker.so has no RPATH; keep the build-tree and install dirs on the loader path
        set_target_properties(${_aker_exe} PROPERTIES
            BUILD_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
            INSTALL_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
        )
    endforeach()
    message(STATUS "Aker: building aker_benchmark_diskann aker_approx_hit_probe against ${AKER_LIBRARY}")
else()
    message(STATUS "Aker: skipping aker_benchmark_diskann (build Aker with scripts/aker/build.sh first)")
endif()

# Find Boost components needed for SPTAG client (if not already found)
if(NOT TARGET Boost::system)
    find_package(Boost COMPONENTS system thread serialization filesystem QUIET)
endif()

# Native C++ QVCache + pgvector (libpq) search-workload
find_package(PostgreSQL)
if(PostgreSQL_FOUND)
    add_executable(qvcache_benchmark_pgvector qvcache/qvcache_benchmark_pgvector.cpp)
    target_link_libraries(qvcache_benchmark_pgvector
        PRIVATE
            greator
            diskann
            qvcache
            TBB::tbb
            OpenMP::OpenMP_CXX
            Boost::program_options
            spdlog::spdlog
            PostgreSQL::PostgreSQL
    )
    target_include_directories(qvcache_benchmark_pgvector
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/backends
            ${PROJECT_SOURCE_DIR}/backends/Greator/include
            ${PROJECT_SOURCE_DIR}/external/DiskANN/include
            ${PostgreSQL_INCLUDE_DIRS}
    )
    add_executable(qvcache_update_pgvector qvcache/qvcache_update_pgvector.cpp)
    target_link_libraries(qvcache_update_pgvector
        PRIVATE
            greator
            diskann
            qvcache
            TBB::tbb
            OpenMP::OpenMP_CXX
            Boost::program_options
            spdlog::spdlog
            PostgreSQL::PostgreSQL
    )
    target_include_directories(qvcache_update_pgvector
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/backends
            ${PROJECT_SOURCE_DIR}/backends/Greator/include
            ${PROJECT_SOURCE_DIR}/external/DiskANN/include
            ${PROJECT_SOURCE_DIR}/benchmarks
            ${PostgreSQL_INCLUDE_DIRS}
    )
    add_executable(qvcache_refresh_pgvector qvcache/qvcache_refresh_pgvector.cpp)
    target_link_libraries(qvcache_refresh_pgvector
        PRIVATE
            greator
            diskann
            qvcache
            TBB::tbb
            OpenMP::OpenMP_CXX
            Boost::program_options
            spdlog::spdlog
            PostgreSQL::PostgreSQL
    )
    target_include_directories(qvcache_refresh_pgvector
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/backends
            ${PROJECT_SOURCE_DIR}/backends/Greator/include
            ${PROJECT_SOURCE_DIR}/external/DiskANN/include
            ${PROJECT_SOURCE_DIR}/benchmarks
            ${PostgreSQL_INCLUDE_DIRS}
    )
    message(STATUS "Building qvcache_benchmark_pgvector, qvcache_update_pgvector, and qvcache_refresh_pgvector against ${PostgreSQL_LIBRARIES}")
    if(AKER_LIBRARY AND EXISTS "${AKER_ROOT}/inc/ak_anns_cache_c_wrapper.h")
        add_executable(aker_benchmark_pgvector aker/aker_benchmark_pgvector.cpp)
        target_link_libraries(aker_benchmark_pgvector
            PRIVATE
                greator
                diskann
                ${AKER_LIBRARY}
                TBB::tbb
                OpenMP::OpenMP_CXX
                Boost::program_options
                spdlog::spdlog
                PostgreSQL::PostgreSQL
        )
        target_include_directories(aker_benchmark_pgvector
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
                ${PROJECT_SOURCE_DIR}/backends
                ${PROJECT_SOURCE_DIR}/backends/Greator/include
                ${PROJECT_SOURCE_DIR}/external/DiskANN/include
                ${AKER_ROOT}/inc
                ${PROJECT_SOURCE_DIR}/benchmarks/aker
                ${PostgreSQL_INCLUDE_DIRS}
        )
        set_target_properties(aker_benchmark_pgvector PROPERTIES
            BUILD_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
            INSTALL_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
        )
        add_executable(aker_update_pgvector aker/aker_update_pgvector.cpp)
        target_link_libraries(aker_update_pgvector
            PRIVATE
                greator
                diskann
                ${AKER_LIBRARY}
                TBB::tbb
                OpenMP::OpenMP_CXX
                Boost::program_options
                spdlog::spdlog
                PostgreSQL::PostgreSQL
        )
        target_include_directories(aker_update_pgvector
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
                ${PROJECT_SOURCE_DIR}/backends
                ${PROJECT_SOURCE_DIR}/backends/Greator/include
                ${PROJECT_SOURCE_DIR}/external/DiskANN/include
                ${AKER_ROOT}/inc
                ${PROJECT_SOURCE_DIR}/benchmarks
                ${PROJECT_SOURCE_DIR}/benchmarks/aker
                ${PostgreSQL_INCLUDE_DIRS}
        )
        set_target_properties(aker_update_pgvector PROPERTIES
            BUILD_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
            INSTALL_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
        )
        add_executable(aker_refresh_pgvector aker/aker_refresh_pgvector.cpp)
        target_link_libraries(aker_refresh_pgvector
            PRIVATE
                greator
                diskann
                ${AKER_LIBRARY}
                TBB::tbb
                OpenMP::OpenMP_CXX
                Boost::program_options
                spdlog::spdlog
                PostgreSQL::PostgreSQL
        )
        target_include_directories(aker_refresh_pgvector
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
                ${PROJECT_SOURCE_DIR}/backends
                ${PROJECT_SOURCE_DIR}/backends/Greator/include
                ${PROJECT_SOURCE_DIR}/external/DiskANN/include
                ${AKER_ROOT}/inc
                ${PROJECT_SOURCE_DIR}/benchmarks
                ${PROJECT_SOURCE_DIR}/benchmarks/aker
                ${PostgreSQL_INCLUDE_DIRS}
        )
        set_target_properties(aker_refresh_pgvector PROPERTIES
            BUILD_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
            INSTALL_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
        )
        message(STATUS "Aker: building aker_benchmark_pgvector, aker_update_pgvector, and aker_refresh_pgvector against ${AKER_LIBRARY} + libpq")
    endif()
else()
    message(STATUS "PostgreSQL/libpq not found; skipping qvcache_benchmark_pgvector")
endif()

# Special handling for qvcache_benchmark_sptag
if(TARGET qvcache_benchmark_sptag)
    target_sources(qvcache_benchmark_sptag PRIVATE ${PROJECT_SOURCE_DIR}/backends/sptag_backend.cpp)
    target_include_directories(qvcache_benchmark_sptag
        PRIVATE
            ${PROJECT_SOURCE_DIR}/backends/Greator/include # For set_sector_len
    )
endif()

# Special handling for backend_benchmark_sptag
if(TARGET backend_benchmark_sptag)
    target_sources(backend_benchmark_sptag PRIVATE ${PROJECT_SOURCE_DIR}/backends/sptag_backend.cpp)
    target_include_directories(backend_benchmark_sptag
        PRIVATE
            ${PROJECT_SOURCE_DIR}/backends
    )
endif()

# Special handling for windowed_backend_benchmark_sptag
if(TARGET windowed_backend_benchmark_sptag)
    target_sources(windowed_backend_benchmark_sptag PRIVATE ${PROJECT_SOURCE_DIR}/backends/sptag_backend.cpp)
    target_include_directories(windowed_backend_benchmark_sptag
        PRIVATE
            ${PROJECT_SOURCE_DIR}/backends
    )
endif()

# Special handling for windowed_qvcache_benchmark_sptag
if(TARGET windowed_qvcache_benchmark_sptag)
    target_sources(windowed_qvcache_benchmark_sptag PRIVATE ${PROJECT_SOURCE_DIR}/backends/sptag_backend.cpp)
    target_include_directories(windowed_qvcache_benchmark_sptag
        PRIVATE
            ${PROJECT_SOURCE_DIR}/backends/Greator/include # For set_sector_len
    )
endif()

# Special handling for compute_groundtruth (simpler, only needs OpenMP)
add_executable(compute_groundtruth compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth
    PRIVATE
        OpenMP::OpenMP_CXX
)
target_include_directories(compute_groundtruth
    PRIVATE
        ${PROJECT_SOURCE_DIR}/external/DiskANN/include
)
