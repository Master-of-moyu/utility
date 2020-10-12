
#find package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#####################################################################
add_executable(feature_detection)

target_sources(feature_detection
    PRIVATE
        feature_detection_test.cpp
        cvmat.cpp
        lkopticalflow.cpp
)

target_compile_features(feature_detection PRIVATE cxx_std_14)
target_link_libraries(feature_detection ${OpenCV_LIBS})

#####################################################################
add_executable(optical_flow)

target_sources(optical_flow
    PRIVATE
        optical_flow_test.cpp
        cvmat.cpp
        lkopticalflow.cpp
)

target_compile_features(optical_flow PRIVATE cxx_std_14)
target_link_libraries(optical_flow ${OpenCV_LIBS})
