set(SOURCE_FILES)

# add_custom_target(proto_files DEPENDS ${PROTO_HDRS})

# function(add_library targetName)
#   _add_library(${targetName} ${ARGN})
#   add_dependencies(${targetName} proto_files)
# endfunction()

if (BUILD_VIDEO_EVALUATORS)
  list(APPEND SOURCE_FILES
    video/decoder_evaluator.cpp
    video/encoder_evaluator.cpp)
endif()

if (BUILD_IMAGE_EVALUATORS)
  list(APPEND SOURCE_FILES
    image/image_decoder_evaluator.cpp)
endif()

if (BUILD_CAFFE_EVALUATORS)
  list(APPEND SOURCE_FILES
    caffe/net_descriptor.cpp
    caffe/caffe_evaluator.cpp)
  if (BUILD_CAFFE_INPUT_EVALUATORS)
    list(APPEND SOURCE_FILES
      caffe/default/default_input_evaluator.cpp
      caffe/yolo/yolo_input_evaluator.cpp
      caffe/facenet/facenet_parser_evaluator.cpp
      caffe/facenet/facenet_input_evaluator.cpp
      caffe/faster_rcnn/faster_rcnn_parser_evaluator.cpp
      caffe/cpm/cpm_person_input_evaluator.cpp
      caffe/cpm/cpm_person_parser_evaluator.cpp
      caffe/cpm/cpm_input_evaluator.cpp
      caffe/cpm/cpm_parser_evaluator.cpp)
    if (BUILD_CPM2_EVALUATORS)
      list(APPEND SOURCE_FILES
        caffe/cpm2/cpm2_input_evaluator.cpp
        caffe/cpm2/cpm2_parser_evaluator.cpp)
    endif()
  endif()
endif()

if (BUILD_IMAGE_PROCESSING_EVALUATORS)
  list(APPEND SOURCE_FILES
    image_processing/blur_evaluator.cpp)
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  list(APPEND SOURCE_FILES
    # movie_analysis/movie_evaluator.cpp
    movie_analysis/histogram_evaluator.cpp
    # movie_analysis/face_evaluator.cpp
    movie_analysis/optical_flow_evaluator.cpp
    #movie_analysis/camera_motion_evaluator.cpp)
    )
endif()

if (BUILD_TRACKER_EVALUATORS)
  list(APPEND SOURCE_FILES
    tracker/tracker_evaluator.cpp)
endif()

if (BUILD_UTIL_EVALUATORS)
  list(APPEND SOURCE_FILES
    util/discard_evaluator.cpp
    util/swizzle_evaluator.cpp)
endif()

add_library(evaluators OBJECT
  ${SOURCE_FILES})
