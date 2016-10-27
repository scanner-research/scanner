set(SOURCE_FILES)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
  types.proto)

list(APPEND SOURCE_FILES
  ${PROTO_SRCS})

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

if (BUILD_CAFFE_EVALUATORS)
  list(APPEND SOURCE_FILES
    caffe/net_descriptor.cpp
    caffe/caffe_evaluator.cpp)
  if (BUILD_CAFFE_INPUT_TRANSFORMERS)
    list(APPEND SOURCE_FILES
      caffe/vgg/vgg_cpu_input_transformer.cpp
      caffe/yolo/yolo_input_evaluator.cpp
      caffe/squeezenet/squeezenet_input_evaluator.cpp
      caffe/facenet/facenet_parser_evaluator.cpp
      caffe/facenet/facenet_input_evaluator.cpp)
  endif()
endif()

if (BUILD_IMAGE_PROCESSING_EVALUATORS)
  list(APPEND SOURCE_FILES
    image_processing/blur_evaluator.cpp)
endif()

if (BUILD_MOVIE_ANALYSIS_EVALUATORS)
  list(APPEND SOURCE_FILES
    # movie_analysis/movie_evaluator.cpp
    # movie_analysis/histogram_evaluator.cpp
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
    util/swizzle_evaluator.cpp)
endif()

add_library(evaluators OBJECT
  ${SOURCE_FILES})
