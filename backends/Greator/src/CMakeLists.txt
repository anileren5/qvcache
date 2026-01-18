find_package(TBB REQUIRED)

set(CPP_SOURCES
    ann_exception.cpp aux_utils.cpp index.cpp
    linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
    partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
    utils.cpp sector_len.cpp v2/graph_delta.cpp v2/index_merger.cpp 
    v2/merge_insert.cpp
)

add_library(greator STATIC ${CPP_SOURCES})

# Include dirs
target_include_directories(greator
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/greator
        /usr/include/mkl
        ${BOOST_ROOT}
)

# Compile definitions
target_compile_definitions(greator
    PUBLIC
        MKL_ILP64
)

# Link libraries
target_link_libraries(greator
    PUBLIC
        TBB::tbb
        mkl_intel_lp64
        mkl_core
        mkl_sequential
        pthread
        m
        dl
        aio
)
