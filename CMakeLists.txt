cmake_minimum_required(VERSION 3.31)
project(Iafps)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

add_executable(Iafps main.cpp)

set_target_properties(Iafps PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(Iafps PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/matplotlib-cpp-master
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_compile_definitions(Iafps PRIVATE
    ${TORCH_CXX_FLAGS}
    WITHOUT_NUMPY
)

target_link_libraries(Iafps
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Python3::Python
    Python3::NumPy
)