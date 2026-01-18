find_package(TBB REQUIRED)

set(CPP_SOURCES abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
    distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
    linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
    in_mem_data_store.cpp in_mem_graph_store.cpp
    natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
    pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)

add_library(diskann STATIC ${CPP_SOURCES})

# Add -fPIC for shared library linking (needed for Python module)
set_target_properties(diskann PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Include dirs
target_include_directories(diskann
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/diskann
        /usr/include/mkl
        ${BOOST_ROOT}
)

# Compile definitions
target_compile_definitions(diskann
    PUBLIC
        MKL_ILP64
)

# Link libraries
target_link_libraries(diskann
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
