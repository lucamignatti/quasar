# Set the minimum CMake version required
cmake_minimum_required(VERSION 3.15)

# Define your project's name and specify it's a C++ project
project(quasar CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)

# --- Add Your Include Directory ---
# This tells your executable where to find its headers (like "rlenv.h")
target_include_directories(quasar PRIVATE
    include
    lib/RocketSim/src
)

# --- Link RocketSim to Your Executable ---
# This links your 'quasar' against the 'RocketSim' library.
# CMake will automatically handle all the include directories and linker flags.
target_link_libraries(quasar PRIVATE RocketSim)
