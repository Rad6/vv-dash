cmake_minimum_required(VERSION 3.4...3.18)
project(istream_renderer)

set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)

pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libswscale
    libavutil
)

add_subdirectory(extern/pybind11)

pybind11_add_module(istream_renderer src/istream_module.cpp)

target_link_libraries(istream_renderer PUBLIC PkgConfig::LIBAV)
target_link_libraries(istream_renderer PUBLIC glfw)
target_link_libraries(istream_renderer PUBLIC ${OPENGL_LIBRARIES})


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(istream_renderer
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})