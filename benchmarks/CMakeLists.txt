set(_BENCH_INCLUDES
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/backends
    ${PROJECT_SOURCE_DIR}/backends/Greator/include
    ${PROJECT_SOURCE_DIR}/external/DiskANN/include
    ${PROJECT_SOURCE_DIR}/benchmarks
)

set(_BENCH_LIBS
    greator
    diskann
    TBB::tbb
    OpenMP::OpenMP_CXX
    Boost::program_options
    spdlog::spdlog
)

function(qvcache_bench name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE ${_BENCH_LIBS} qvcache ${ARGN})
    target_include_directories(${name} PRIVATE ${_BENCH_INCLUDES})
endfunction()

function(aker_bench name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE ${_BENCH_LIBS} ${AKER_LIBRARY} ${ARGN})
    target_include_directories(${name} PRIVATE
        ${_BENCH_INCLUDES}
        ${AKER_ROOT}/inc
        ${PROJECT_SOURCE_DIR}/benchmarks/aker
    )
    set_target_properties(${name} PROPERTIES
        BUILD_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
        INSTALL_RPATH "${AKER_ROOT}/build/lib;/usr/local/lib;/opt/boost_1_86/lib"
    )
endfunction()

qvcache_bench(diskann_build_index diskann/build_index.cpp)

qvcache_bench(qvcache_diskann_search qvcache/diskann_search.cpp)

set(AKER_ROOT "${PROJECT_SOURCE_DIR}/external/Aker")
find_library(AKER_LIBRARY
    NAMES aker
    PATHS "${AKER_ROOT}/build/lib" /usr/local/lib
)
if(AKER_LIBRARY AND EXISTS "${AKER_ROOT}/inc/ak_anns_cache_c_wrapper.h")
    aker_bench(aker_diskann_search aker/diskann_search.cpp)
    message(STATUS "Aker: building aker_diskann_search against ${AKER_LIBRARY}")
else()
    message(STATUS "Aker: skipping aker_diskann_search (build Aker with scripts/aker/build.sh first)")
endif()

find_package(PostgreSQL)
if(PostgreSQL_FOUND)
    qvcache_bench(qvcache_pgvector_search qvcache/pgvector_search.cpp PostgreSQL::PostgreSQL)
    qvcache_bench(qvcache_pgvector_refresh qvcache/pgvector_refresh.cpp PostgreSQL::PostgreSQL)
    message(STATUS "Building qvcache_pgvector_search and qvcache_pgvector_refresh against ${PostgreSQL_LIBRARIES}")
    if(AKER_LIBRARY AND EXISTS "${AKER_ROOT}/inc/ak_anns_cache_c_wrapper.h")
        aker_bench(aker_pgvector_search aker/pgvector_search.cpp PostgreSQL::PostgreSQL)
        aker_bench(aker_pgvector_refresh aker/pgvector_refresh.cpp PostgreSQL::PostgreSQL)
        message(STATUS "Aker: building aker_pgvector_search and aker_pgvector_refresh against ${AKER_LIBRARY} + libpq")
    endif()
else()
    message(STATUS "PostgreSQL/libpq not found; skipping pgvector benchmarks")
endif()

add_executable(compute_groundtruth workload/compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(compute_groundtruth
    PRIVATE
        ${PROJECT_SOURCE_DIR}/external/DiskANN/include
)
