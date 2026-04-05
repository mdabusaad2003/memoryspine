cmake_minimum_required(VERSION 3.14)
project(MemorySpine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the shared library
add_library(memspine SHARED memspine.cpp memspine.h)

# Add options for slot counts / dims overrides if desired
option(SPINE_DIM "Embedding dimensions" 768)
option(SPINE_SLOTS "Maximum number of slots" 10000000)

target_compile_definitions(memspine PRIVATE
    SPINE_DIM=${SPINE_DIM}
    SPINE_SLOTS=${SPINE_SLOTS}
)

# No Curl dependency needed for the Python wrapped version
if(WIN32)
    # Windows doesn't need fPIC, MSVC handles it with SHARED
else()
    target_compile_options(memspine PRIVATE -fPIC -O3)
endif()

# Compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(memspine PRIVATE /O2)
    else()
        target_compile_options(memspine PRIVATE -O2 -march=native)
    endif()
endif()
