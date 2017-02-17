# TODO: set up a FindOpenFace.cmake
set(OpenFace_DIR "/home/wcrichto/.deps/OpenFace")

set(SOURCE_FILES openface_kernel.cpp)

add_library(openface OBJECT ${SOURCE_FILES})

target_include_directories(openface PUBLIC
  "${OpenFace_DIR}/lib/local/LandmarkDetector/include"
  "${OpenFace_DIR}/lib/local/FaceAnalyser/include"
  "${OpenFace_DIR}/lib/3rdParty/dlib/include")
list(APPEND STDLIB_LIBRARIES
  "${OpenFace_DIR}/build/lib/local/LandmarkDetector/libLandmarkDetector.a"
  "${OpenFace_DIR}/build/lib/local/FaceAnalyser/libFaceAnalyser.a"
  "${OpenFace_DIR}/build/lib/3rdParty/dlib/libdlib.a")

set(STDLIB_LIBRARIES ${STDLIB_LIBRARIES} PARENT_SCOPE)
