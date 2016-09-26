#include "JPEGWriter.h"

// We need version 6b
#if JPEG_LIB_VERSION < 62
#error JPEGWriter needs IJG libjpeg with a version of at least 6b.
#endif

#include <cassert>
#include <stdexcept>

namespace {
// Bounce errors to the object
void libjpeg_error_exit(j_common_ptr cinfo) {
  printf("is decom: %d, client_dat: %lld\n", cinfo->is_decompressor,
         (long long)cinfo->client_data);
  assert(cinfo->is_decompressor && cinfo->client_data);
  ((JPEGWriter*)cinfo->client_data)->error_exit();
}

void libjpeg_output_message(j_common_ptr cinfo) {
  assert(!cinfo->is_decompressor && cinfo->client_data);
  ((JPEGWriter*)cinfo->client_data)->output_message();
}
}

JPEGWriter::JPEGWriter() {
  // Error handling first, in case the initialization fails.
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = libjpeg_error_exit;
  jerr.output_message = libjpeg_output_message;

  // Initialize the compression part.
  jpeg_create_compress(&cinfo);
  assert(sizeof(JSAMPLE) == 1);

  cinfo.client_data = this;
}

JPEGWriter::~JPEGWriter() {
  assert(cinfo.client_data == this);
  jpeg_destroy_compress(&cinfo);
}

void JPEGWriter::header(const unsigned width, const unsigned height,
                        const unsigned components,
                        const JPEG::ColorSpace colorSpace) {
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = components;
  cinfo.in_color_space = (J_COLOR_SPACE)colorSpace;
  jpeg_set_defaults(&cinfo);
}

void JPEGWriter::setTradeoff(const JPEG::TimeQualityTradeoff value) {
  switch (value) {
    case JPEG::FASTER:
      cinfo.dct_method = JDCT_FASTEST;
      break;

    case JPEG::DEFAULT:
      cinfo.dct_method = JDCT_DEFAULT;
      break;

    case JPEG::BETTER:
      cinfo.dct_method = JDCT_FLOAT;
      break;

    default:
      assert(!"Invalid time/quality tradeoff.");
      break;
  }
}

void JPEGWriter::error_exit() {
  output_message();
  throw std::runtime_error("libjpeg error: " + warningMsg);
}

void JPEGWriter::output_message() {
  // Use the default routine to generate the message.
  char buffer[JMSG_LENGTH_MAX];
  (cinfo.err->format_message)((jpeg_common_struct*)&cinfo, buffer);

  warningMsg += buffer;
  warningMsg += '\n';
}
