CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)

project(Exercises)

find_package(OpenCV REQUIRED)

add_executable(exercise_01 src/exercise_01.cpp)
add_executable(exercise_02 src/exercise_02.cpp)
add_executable(exercise_03 src/exercise_03.cpp)
add_executable(exercise_04 src/exercise_04.cpp)
add_executable(exercise_05 src/exercise_05.cpp)

target_link_libraries(exercise_01 ${OpenCV_LIBS})
target_link_libraries(exercise_02 ${OpenCV_LIBS})
target_link_libraries(exercise_03 ${OpenCV_LIBS})
target_link_libraries(exercise_04 ${OpenCV_LIBS})
target_link_libraries(exercise_05 ${OpenCV_LIBS})