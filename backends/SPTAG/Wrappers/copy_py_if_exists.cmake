# CMake script to copy Python file if it exists
# Usage: cmake -P copy_py_if_exists.cmake <source> <destination>
if(ARGC LESS 2)
    message(FATAL_ERROR "Usage: cmake -P copy_py_if_exists.cmake <source> <destination>")
endif()

set(SOURCE_FILE "${ARGV0}")
set(DEST_FILE "${ARGV1}")

# Resolve to absolute paths
get_filename_component(SOURCE_FILE "${SOURCE_FILE}" ABSOLUTE)
get_filename_component(DEST_FILE "${DEST_FILE}" ABSOLUTE)

message(STATUS "Looking for Python file: ${SOURCE_FILE}")

if(EXISTS "${SOURCE_FILE}")
    get_filename_component(DEST_DIR "${DEST_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${DEST_DIR}")
    get_filename_component(DEST_NAME "${DEST_FILE}" NAME)
    # Use configure_file with COPYONLY to preserve the file exactly
    configure_file("${SOURCE_FILE}" "${DEST_DIR}/${DEST_NAME}" COPYONLY)
    message(STATUS "Copied ${SOURCE_FILE} to ${DEST_FILE}")
else()
    message(FATAL_ERROR "Python file ${SOURCE_FILE} not found! SWIG must have failed to generate it. This is required for the Python client to work.")
endif()
