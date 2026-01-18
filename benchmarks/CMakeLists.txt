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

# Find Boost components needed for SPTAG client (if not already found)
if(NOT TARGET Boost::system)
    find_package(Boost COMPONENTS system thread serialization filesystem QUIET)
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
