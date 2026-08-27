cmake_minimum_required(VERSION 3.21)

project(neural_network LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

# Source collection

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE SOURCE_FILES
    CONFIGURE_DEPENDS
    ${SRC_DIR}/*.cpp
)

# Remove Main.cpp from library sources
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*Main\\.cpp$")

file(GLOB_RECURSE ALL_PATHS
    LIST_DIRECTORIES true
    CONFIGURE_DEPENDS
    ${SRC_DIR}/*
)

set(INCLUDE_DIRS ${SRC_DIR})

foreach(path ${ALL_PATHS})
    if(IS_DIRECTORY ${path})
        list(APPEND INCLUDE_DIRS ${path})
    endif()
endforeach()

list(REMOVE_DUPLICATES INCLUDE_DIRS)

# Core library

add_library(neural_network_core ${SOURCE_FILES})

target_include_directories(neural_network_core
    PUBLIC
    ${INCLUDE_DIRS}
    ${CURSES_INCLUDE_DIRS}
)


# Main executable
add_executable(neural_network
    ${SRC_DIR}/Main.cpp
)

target_link_libraries(neural_network
    PRIVATE
    neural_network_core
)

