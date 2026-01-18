# Utils executables for converting vector formats

# ivecs_to_bin: Convert ivecs format to binary format
add_executable(ivecs_to_bin ivecs_to_bin.cpp)

target_include_directories(ivecs_to_bin
    PRIVATE
        ${PROJECT_SOURCE_DIR}/utils
        ${PROJECT_SOURCE_DIR}/external/DiskANN/include
)

# fvecs_to_bin: Convert fvecs format to binary format
add_executable(fvecs_to_bin fvecs_to_bin.cpp)

target_include_directories(fvecs_to_bin
    PRIVATE
        ${PROJECT_SOURCE_DIR}/utils
        ${PROJECT_SOURCE_DIR}/external/DiskANN/include
)
