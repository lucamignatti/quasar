cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(quasar CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(ENABLE_TRACING "Enable performance tracing" OFF)
option(LIBTORCH_USE_CXX11_ABI "Set _GLIBCXX_USE_CXX11_ABI for libtorch compatibility (0 or 1). Leave empty to not set." "")

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
    src/env/rlenv.cpp
    src/env/vecenv.cpp
    src/rl/MLP.cpp
    src/rl/PPO.cpp
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

# Always link RocketSim
target_link_libraries(quasar PRIVATE RocketSim)

set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
message(STATUS "Using libtorch from: ${LIBTORCH_DIR}")

find_package(Torch REQUIRED)
message(STATUS "libtorch: ${TORCH_VERSION}")

if(DEFINED TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS "Applied TORCH_CXX_FLAGS to CMAKE_CXX_FLAGS")
endif()

if(TARGET Torch::Torch)
    target_link_libraries(quasar PRIVATE Torch::Torch)
else()
    target_link_libraries(quasar PRIVATE "${TORCH_LIBRARIES}")
endif()

if (MSVC)
  if(DEFINED TORCH_INSTALL_PREFIX)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  else()
    # Fall back to common variable used by some Torch CMake configs
    file(GLOB TORCH_DLLS "${TORCH_ROOT}/lib/*.dll" "${TORCH_HOME}/lib/*.dll")
  endif()
  add_custom_command(TARGET quasar
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:quasar>)
endif()

if(LIBTORCH_USE_CXX11_ABI)
    # Set per-target compile definition to control C++ ABI compatibility
    target_compile_definitions(quasar PRIVATE _GLIBCXX_USE_CXX11_ABI=${LIBTORCH_USE_CXX11_ABI})
endif()

target_compile_definitions(quasar PRIVATE USE_LIBTORCH=1)
