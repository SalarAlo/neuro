cmake_minimum_required(VERSION 3.21)

project(neural_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use Debug by default for single-configuration generators.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(PROJECT_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
)

# Source collection

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE SOURCE_FILES
    CONFIGURE_DEPENDS
    "${SRC_DIR}/*.cpp"
)

# Main.cpp belongs only to the executable.
list(FILTER SOURCE_FILES EXCLUDE REGEX "[/\\\\]Main\\.cpp$")

file(GLOB_RECURSE ALL_PATHS
    LIST_DIRECTORIES true
    CONFIGURE_DEPENDS
    "${SRC_DIR}/*"
)

set(INCLUDE_DIRS "${SRC_DIR}")

foreach(path IN LISTS ALL_PATHS)
    if(IS_DIRECTORY "${path}")
        list(APPEND INCLUDE_DIRS "${path}")
    endif()
endforeach()

list(REMOVE_DUPLICATES INCLUDE_DIRS)

# Core library

if(SOURCE_FILES)
    add_library(neural_network_core ${SOURCE_FILES})

    target_include_directories(neural_network_core
        PUBLIC
            ${INCLUDE_DIRS}
    )

    target_compile_options(neural_network_core
        PRIVATE
            ${PROJECT_WARNING_FLAGS}
    )
else()
    # Header-only until the project contains non-Main.cpp source files.
    add_library(neural_network_core INTERFACE)

    target_include_directories(neural_network_core
        INTERFACE
            ${INCLUDE_DIRS}
    )
endif()

# Main executable

add_executable(neural_network
    "${SRC_DIR}/Main.cpp"
)

target_link_libraries(neural_network
    PRIVATE
        neural_network_core
)

target_compile_options(neural_network
    PRIVATE
        ${PROJECT_WARNING_FLAGS}

        $<$<CONFIG:Release>:
            -O3
            -march=native
            -DNDEBUG
        >
)

