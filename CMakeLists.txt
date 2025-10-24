cmake_minimum_required(VERSION 3.15)

project(quasar CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(ENABLE_TRACING "Enable performance tracing (chrome://tracing compatible)" OFF)

if(ENABLE_TRACING)
    message(STATUS "Performance tracing: ENABLED")
else()
    message(STATUS "Performance tracing: DISABLED (maximum performance)")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        # Aggressive optimization for Release builds
        add_compile_options(-O3 -march=native -DNDEBUG -flto)
        add_link_options(-flto)
        message(STATUS "Compiler optimizations: -O3 -march=native -flto")
    elseif(MSVC)
        add_compile_options(/O2 /DNDEBUG /GL)
        add_link_options(/LTCG)
        message(STATUS "Compiler optimizations: /O2 /GL /LTCG")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-O0 -g)
    elseif(MSVC)
        add_compile_options(/Od /Zi)
    endif()
endif()

add_subdirectory(lib/RocketSim)

add_executable(quasar
    src/main.cpp
    src/rlenv.cpp
    src/vecenv.cpp
)

target_include_directories(quasar PRIVATE
    include
    lib/RocketSim/src
)

if(ENABLE_TRACING)
    target_compile_definitions(quasar PRIVATE ENABLE_TRACING=1)
else()
    target_compile_definitions(quasar PRIVATE ENABLE_TRACING=0)
endif()

target_link_libraries(quasar PRIVATE RocketSim)
