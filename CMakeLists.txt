cmake_minimum_required(VERSION 3.11)
project(test)

add_executable(dtw_test)
target_sources(dtw_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/DTW/dtw_distance.cpp)
target_compile_features(dtw_test PRIVATE cxx_std_14)
