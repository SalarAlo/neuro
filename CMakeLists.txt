cmake_minimum_required(VERSION 3.21)

project(neuro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(PROJECT_IS_TOP_LEVEL)

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

# Core library

if(SOURCE_FILES)
    add_library(neuro_core ${SOURCE_FILES})

    target_include_directories(neuro_core
        PUBLIC
            "${SRC_DIR}"
    )
    target_compile_features(neuro_core PUBLIC cxx_std_23)

    target_compile_options(neuro_core
        PRIVATE
            ${PROJECT_WARNING_FLAGS}
            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-Og>
    )
else()
    # Header-only until the project contains non-Main.cpp source files.
    add_library(neuro_core INTERFACE)

    target_include_directories(neuro_core
        INTERFACE
            "${SRC_DIR}"
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
            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-Og>

            $<$<CONFIG:Release>:
                -O3
                -march=native
                -DNDEBUG
            >
    )
endif()

