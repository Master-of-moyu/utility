
add_executable(dtw_test)

target_sources(dtw_test PRIVATE dtw_distance.cpp)

target_compile_features(dtw_test PRIVATE cxx_std_14)
