CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)

project(Exercises)

find_package(OpenCV REQUIRED)

add_executable(exercise_03 src/exercise_03.cpp)
add_executable(exercise_04 src/exercise_04.cpp)

target_link_libraries(exercise_03 ${OpenCV_LIBS})
target_link_libraries(exercise_04 ${OpenCV_LIBS})