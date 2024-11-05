cmake_minimum_required(VERSION 3.0)
project(face LANGUAGES CXX)

add_executable(${PROJECT_NAME} main.cpp detect.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
include_directories(${PROJECT_NAME} PRIVATE include)

find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

