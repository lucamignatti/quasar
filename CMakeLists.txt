# Set the minimum CMake version required
cmake_minimum_required(VERSION 3.15)

# Define your project's name and specify it's a C++ project
project(quasar CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build Configuration ---
# Default to Release build for maximum performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# --- Tracing Option ---
# Option to enable/disable performance tracing (default: OFF for speed)
option(ENABLE_TRACING "Enable performance tracing (chrome://tracing compatible)" OFF)

if(ENABLE_TRACING)
    message(STATUS "Performance tracing: ENABLED")
else()
    message(STATUS "Performance tracing: DISABLED (maximum performance)")
endif()

# --- Compiler Optimization Flags ---
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

# --- Add RocketSim ---
# This line tells CMake to look in the 'lib/RocketSim' folder
# and build its 'CMakeLists.txt' file.
add_subdirectory(lib/RocketSim)

# --- Add Your Executable ---
# This creates your program, named 'quasar',
# from all your source files.
add_executable(quasar
    src/main.cpp
    src/rlenv.cpp
    src/vecenv.cpp
)

# --- Add Your Include Directory ---
# This tells your executable where to find its headers (like "rlenv.h")
target_include_directories(quasar PRIVATE
    include
    lib/RocketSim/src
)

# --- Tracing Configuration ---
if(ENABLE_TRACING)
    target_compile_definitions(quasar PRIVATE ENABLE_TRACING=1)
else()
    target_compile_definitions(quasar PRIVATE ENABLE_TRACING=0)
endif()

# --- Link RocketSim to Your Executable ---
# This links your 'quasar' against the 'RocketSim' library.
# CMake will automatically handle all the include directories and linker flags.
target_link_libraries(quasar PRIVATE RocketSim)
