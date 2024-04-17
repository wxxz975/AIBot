
cmake_minimum_required(VERSION 3.10)

project(AIBot)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenVINO COMPONENTS Runtime)
include_directories(${OpenVINO_INCLUDE_DIRS}/include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(
    ${PROJECT_NAME}
    src/main.cpp
    )

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    openvino::runtime
)