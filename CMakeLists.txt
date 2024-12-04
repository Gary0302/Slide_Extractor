cmake_minimum_required(VERSION 3.10)
project(SlideExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable
add_executable(slide_extractor slide_extractor.cpp)

# Link OpenCV libraries
target_link_libraries(slide_extractor ${OpenCV_LIBS})