cmake_minimum_required(VERSION 3.31)
project(Iafps)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_calib3d OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_dnn OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_features2d OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_flann OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_gapi OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_ml OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_photo OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python_tests OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_stitching OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_ts OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_video OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_videoio OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python_bindings_generator OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_java_bindings_generator OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_js_bindings_generator OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_objc_bindings_generator OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_java OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_js OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_objc OFF CACHE BOOL "" FORCE)

add_subdirectory(opencv-4.12.0)

add_executable(Iafps main.cpp)

set_target_properties(Iafps PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(Iafps PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.12.0/modules/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.12.0/modules/imgproc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.12.0/modules/imgcodecs/include
    ${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.12.0/modules/highgui/include
    ${CMAKE_CURRENT_SOURCE_DIR}/matplotlib-cpp-master
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_compile_definitions(Iafps PRIVATE
    ${TORCH_CXX_FLAGS}
    WITHOUT_NUMPY
)

add_dependencies(Iafps opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)

target_link_libraries(Iafps
    ${TORCH_LIBRARIES}
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    Python3::Python
    Python3::NumPy
)