cmake_minimum_required(VERSION 3.10)

project(Aker LANGUAGES C CXX)

# -----------------------------
# Basic build settings
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Default build type for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Do not embed build-tree RPATH. Executables must locate libaker.so via the
# system dynamic loader search paths (e.g., ldconfig-managed paths or LD_LIBRARY_PATH).
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "")

# Output directories (under the project root).
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")

# Keep the existing behavior: warnings suppressed by default.
option(AKER_SUPPRESS_WARNINGS "Suppress compiler warnings (-w)" ON)

# Cache threshold policy mode (compile-time)
option(AKER_ENABLE_PROXIMITY_MODE "Enable Proximity Mode (fixed threshold per entry)" OFF)
option(AKER_ENABLE_POTLUCK_MODE "Enable Potluck Mode (global threshold tuning + dropout)" OFF)

if(AKER_ENABLE_PROXIMITY_MODE AND AKER_ENABLE_POTLUCK_MODE)
    message(FATAL_ERROR "AKER_ENABLE_PROXIMITY_MODE and AKER_ENABLE_POTLUCK_MODE are mutually exclusive")
endif()

find_package(Threads REQUIRED)

# -----------------------------
# Boost (shared-only)
# -----------------------------
set(BOOST_ROOT "" CACHE PATH "Boost root directory (optional)")
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost 1.80 REQUIRED COMPONENTS
    program_options
    log
    log_setup
    date_time
    thread
    system
    filesystem
    regex
    atomic
    chrono
)

# -----------------------------
# FAISS
# -----------------------------
set(FAISS_ROOT "" CACHE PATH "FAISS installation prefix")

find_path(FAISS_INCLUDE_DIR
    NAMES faiss/Index.h
    PATHS "${FAISS_ROOT}/include" /usr/local/include /usr/include
    REQUIRED
)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS "${FAISS_ROOT}/lib" "${FAISS_ROOT}/lib64" /usr/local/lib /usr/lib /usr/lib64
    REQUIRED
)

add_library(faiss::faiss UNKNOWN IMPORTED)
set_target_properties(faiss::faiss PROPERTIES
    IMPORTED_LOCATION "${FAISS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
)

# -----------------------------
# Aker shared library (libaker.so)
# -----------------------------
file(GLOB_RECURSE AKER_SRC_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.c"
    "${PROJECT_SOURCE_DIR}/src/*.cc"
)

add_library(aker SHARED ${AKER_SRC_FILES})

target_include_directories(aker
    PUBLIC
        "${PROJECT_SOURCE_DIR}/inc"
        "${PROJECT_SOURCE_DIR}/extern"
)

# Debug symbols and no optimization in Debug.
target_compile_options(aker PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-ggdb>
)

if(AKER_SUPPRESS_WARNINGS)
    target_compile_options(aker PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-w>
    )
else()
    target_compile_options(aker PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

# Compile-time mode selection (exactly one must be enabled by the header contract).
if(AKER_ENABLE_POTLUCK_MODE)
    target_compile_definitions(aker PUBLIC
        AKER_ENABLE_PROXIMITY_MODE=0
        AKER_ENABLE_POTLUCK_MODE=1
        AKER_STANDARD_MODE=0
    )
elseif(AKER_ENABLE_PROXIMITY_MODE)
    target_compile_definitions(aker PUBLIC
        AKER_ENABLE_PROXIMITY_MODE=1
        AKER_ENABLE_POTLUCK_MODE=0
        AKER_STANDARD_MODE=0
    )
else()
    target_compile_definitions(aker PUBLIC
        AKER_ENABLE_PROXIMITY_MODE=0
        AKER_ENABLE_POTLUCK_MODE=0
        AKER_STANDARD_MODE=1
    )
endif()

# Help Boost headers select shared linking where relevant.
target_compile_definitions(aker PRIVATE
    BOOST_LOG_DYN_LINK
    BOOST_LOG_SETUP_DYN_LINK
)

# Link dependencies into libaker.so.
# - Keep these PRIVATE so executables can link only to libaker.so.
target_link_libraries(aker
    PRIVATE
        Threads::Threads
        faiss::faiss
        Boost::program_options
        Boost::log
        Boost::log_setup
        Boost::date_time
        Boost::thread
        Boost::system
        Boost::filesystem
        Boost::regex
        Boost::atomic
        Boost::chrono
)

# -----------------------------
# ak_ycsb_gen (simZipf Zipfian key sequence generator)
# -----------------------------
add_executable(ak_ycsb_gen
    "${PROJECT_SOURCE_DIR}/apps/ak_ycsb_gen.cc"
)

target_include_directories(ak_ycsb_gen
    PRIVATE
        "${PROJECT_SOURCE_DIR}/inc"
        "${PROJECT_SOURCE_DIR}/extern"
)

# Link only to libaker.so. Runtime discovery of libaker.so is via loader search paths.
target_link_libraries(ak_ycsb_gen PRIVATE aker)

# ak_ycsb_gen directly uses Boost.Program_options (CLI parsing). The GNU linker
# does not resolve executable symbols through DT_NEEDED dependencies of a shared
# library by default, so we must link this component explicitly.
target_link_libraries(ak_ycsb_gen PRIVATE Boost::program_options)

# -----------------------------
# aker-random-cache-test (lightweight cache smoke test)
# -----------------------------
add_executable(aker-random-cache-test
    "${PROJECT_SOURCE_DIR}/test/ak_test_random_cache.cc"
)

target_include_directories(aker-random-cache-test
    PRIVATE
        "${PROJECT_SOURCE_DIR}/inc"
        "${PROJECT_SOURCE_DIR}/extern"
)

target_link_libraries(aker-random-cache-test PRIVATE aker)

# -----------------------------
# Install rules
# -----------------------------
include(GNUInstallDirs)

install(TARGETS aker
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/inc/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS aker-random-cache-test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
