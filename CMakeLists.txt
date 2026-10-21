cmake_minimum_required(VERSION 3.21)

project(neuro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Use Release by default for single-configuration generators.
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
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
    add_library(neuro_core ${SOURCE_FILES})

    target_include_directories(neuro_core
        PUBLIC
            ${INCLUDE_DIRS}
    )
    target_compile_features(neuro_core PUBLIC cxx_std_23)

    target_compile_options(neuro_core
        PRIVATE
            ${PROJECT_WARNING_FLAGS}
    )
else()
    # Header-only until the project contains non-Main.cpp source files.
    add_library(neuro_core INTERFACE)

    target_include_directories(neuro_core
        INTERFACE
            ${INCLUDE_DIRS}
    )
    target_compile_features(neuro_core INTERFACE cxx_std_23)
endif()

add_library(Neuro::core ALIAS neuro_core)

# Main executable
if(PROJECT_IS_TOP_LEVEL)
    add_executable(neuro
        "${SRC_DIR}/Main.cpp"
    )

    target_link_libraries(neuro
        PRIVATE
            Neuro::core
    )

    target_compile_options(neuro
        PRIVATE
            ${PROJECT_WARNING_FLAGS}

            $<$<CONFIG:Release>:
                -O3
                -march=native
                -DNDEBUG
            >
    )
endif()

