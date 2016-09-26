#ifndef AJS_JPEG_READER_H
#define AJS_JPEG_READER_H

#include <cassert>
#include <cstdio>
#include <cstdlib>

extern "C" {
#include <jerror.h>
#include <jpeglib.h>
}

#include <string>
#include <vector>

#include "JPEG.h"

/// \class JPEGReader JPEGReader.h
/// Load a JPEG file into a memory buffer using \c libjpeg.
/// Thread-safe if multiple threads do not access the same JPEGReader at once.
/// Loads directly into the caller's memory, minimizing copying.
class JPEGReader {
 public:
  /// Initialize the libjpeg structures.
  JPEGReader();

  /// Free the libjpeg structures.
  ~JPEGReader();

  /// Start the load by reading the header of the JPEG file at \c path.
  /// After this point, width(), height(), components() are all valid.
  void header(const std::string& path);

  void header_mem(uint8_t* data, size_t size);

  /// The current output width of the image.  Valid after calling \c header().
  unsigned width() const;

  /// The current output height of the image.  Valid after calling \c header().
  unsigned height() const;

  /// The current number of components of the image.  Valid after calling \c
  /// header().
  /// \note It is not the case that components() == colorComponents() in all
  /// cases.
  /// When quantizing
  /// to generate an indexed image, components() == 1, but colorComponents()
  /// will
  /// equal the number of components from the color space (usually 1, 3 or 4).
  /// For figuring out buffer sizes to pass to load(), use components.  For
  /// interpreting the color map returned by colormap, use colorComponents().
  unsigned components() const;

  /// The current number of color components of the image.  Valid after calling
  /// \c header().
  /// \sa components()
  unsigned colorComponents() const;

  /// Get the colormap, which can be empty if no quantization has occurred,
  /// or if load() has not yet been called.
  ///
  /// The colormap is stored as colorComponents() * N colors, where N is the
  /// number of actual colors libjpeg was able to reduce to.  The storage is
  /// laid out like so, if we assume for the example that colorComponents() ==
  /// 3.
  /// \code
  /// assert(loader.colorComponents() == 3);
  /// const std::vector<unsigned char>& map = loader.colormap();
  /// const unsigned num_colors = map.size() / 3;
  /// for (unsigned i = 0; i < num_colors; ++i) {
  ///     const unsigned char r = map[3 * i];
  ///     const unsigned char g = map[3 * i + 1];
  ///     const unsigned char b = map[3 * i + 2];
  ///     ...
  /// }
  /// \endcode
  const std::vector<unsigned char>& colormap() const;

  /// Return the current extraction scale.
  JPEG::Scale scale() const;

  /// Set the extraction scale.  Changes width() and height().
  void setScale(const JPEG::Scale value);

  /// Choose a good extraction scale if you ultimately want at most targetWidth
  /// x targetHeight
  /// pixels.  Changes width() and height().
  void chooseGoodScale(const unsigned targetWidth, const unsigned targetHeight);

  /// Return the current extraction colorspace.
  JPEG::ColorSpace colorSpace() const;

  /// Set the extraction colorspace.  Can change components()!
  void setColorSpace(const JPEG::ColorSpace value);

  /// Return the current dithering method if quantizing.
  JPEG::Dither dither() const;

  /// Set the dithering method if we quantize.
  void setDither(const JPEG::Dither& value);

  /// Get the current quantization setting -- zero means no quantization.
  unsigned quantization() const;

  /// Set the current quantization maximum number of colors, or zero for no
  /// quantization.  Can change components()!
  void setQuantization(const unsigned value);

  /// Set the time/quality tradeoff.
  void setTradeoff(const JPEG::TimeQualityTradeoff value);

  /// \name Loading-related member functions.
  /// @{

  /// Get the recommended number of rows to provide.  Valid after calling \c
  /// header().
  unsigned numRecommendedRowPtrs() const;

  /// The maximum number of row pointers to cache in load().
  unsigned maxRowPtrs() const;

  /// Set the maximum number of row pointers to cache in load().
  void setMaxRowPtrs(const unsigned value);

  /// Load the image data into the memory referenced by the row pointer iterator
  /// \c rows.
  ///
  /// A RowPtrIter should act basically like an \c unsigned char**:
  ///
  /// - It must dereference to the start of a chunk of memory at least
  ///   width() * components() bytes long.
  /// - It should be incrementable height() times.
  /// - It will be pre-incremented immediately before each dereference, except
  ///   for the first time.
  /// Normal pointers, iterators into a std::vector<unsigned char*>, and related
  /// things will all work as you expect.
  ///
  /// A normal usage would be something like:
  /// \code
  /// unsigned char** rows = new unsigned char*[loader.height()];
  /// for (int i = 0; i < loader.height(); ++i)
  ///     rows[i] = new unsigned char[loader.width()];
  /// loader.load(rows);
  /// \endcode
  /// Of course, the equivalent code using \c std::vector is highly recommended.
  ///
  /// Furthermore, if you set the maximum number of "live" row pointers to be
  /// used at once using setMaxRowPtrs(), then we guarantee that we will cache
  /// only that many values returned from your RowIter::operator*().
  /// Essentially,
  /// we will feed maxRowPtrs() into libjpeg at once.
  ///
  /// This is useful when RowPtrIter is something more specialized than a real
  /// unsigned char**.  For example, often you don't want to have to allocate
  /// the entire width() * height() pixels at once, but rather only a single
  /// row.  If you do this, obviously you need to do something with the row
  /// before
  /// it gets overwritten with the next row.  You can do this by providing a
  /// specialized version of \c operator++() that does what you want.  Example:
  /// \code
  /// struct OutputRowIter {
  ///     OutputRowIter(std::ostream& out, const unsigned size):
  ///         o(out),
  ///         buffer(size) {}
  ///
  ///     unsigned char* operator*() {
  ///         return &buffer[0];
  ///     }
  ///
  ///     // Write out the current buffer before it gets
  ///     // clobbered by the next row.
  ///     OutputRowIter& operator++() {
  ///         o.write((char*)(&buffer[0]), buffer.size());
  ///         return *this;
  ///     }
  ///
  ///     std::ostream& o;
  ///     std::vector<unsigned char> buffer;
  /// };
  ///
  /// OutputRowIter rowIter(output_file,
  ///                       loader.width() * loader.components());
  /// loader.load(rowIter);
  /// \endcode
  /// Note that this example has a efficiency problem -- iterators should be
  /// cheap to copy, but this one will copy \c buffer...
  template <typename RowPtrIter>
  void load(RowPtrIter rows);

  /// @}

  /// Get warnings generated by libjpeg since the last call to header().
  /// Separate warnings are separated by a newline.
  const std::string& warnings() const;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  /// libjpeg wants us to exit because of an error (private).
  void error_exit();

  /// libjpeg wants us to output a message because of an error (private).
  void output_message();
#endif

 private:
  // Disallow copying
  JPEGReader(const JPEGReader& other);
  JPEGReader& operator=(const JPEGReader& other);

  /// Fill the cache of row pointers.
  template <typename RowIter>
  void fill(RowIter& rows, const unsigned count, bool first_row);

  struct jpeg_decompress_struct cinfo;  /// libjpeg file structure
  struct jpeg_error_mgr jerr;           /// libjpeg error structure

  FILE* file;             /// Good ol' C file IO
  unsigned max_row_ptrs;  /// How many simultaneous rows can the user handle?
  std::vector<unsigned char*> row_ptrs;  /// Cached row pointers from the user

  std::vector<unsigned char>
      cmap;                /// The color map if quantization is requested
  std::string warningMsg;  /// All the warnings.
};

inline unsigned JPEGReader::width() const { return cinfo.output_width; }

inline unsigned JPEGReader::height() const { return cinfo.output_height; }

inline unsigned JPEGReader::components() const {
  return cinfo.output_components;
}

inline unsigned JPEGReader::colorComponents() const {
  return cinfo.out_color_components;
}

inline const std::vector<unsigned char>& JPEGReader::colormap() const {
  return cmap;
}

inline unsigned JPEGReader::numRecommendedRowPtrs() const {
  return cinfo.rec_outbuf_height;
}

inline unsigned JPEGReader::maxRowPtrs() const { return max_row_ptrs; }

inline void JPEGReader::setMaxRowPtrs(const unsigned value) {
  assert(value > 0);
  max_row_ptrs = value;
}

inline const std::string& JPEGReader::warnings() const { return warningMsg; }

template <typename RowIter>
void JPEGReader::fill(RowIter& rows, const unsigned count, bool first_row) {
  for (unsigned i = 0; i < count; ++i) {
    if (!first_row) ++rows;
    row_ptrs[i] = *rows;
    first_row = false;
  }
}

template <typename RowPtrIter>
void JPEGReader::load(RowPtrIter rows) {
  // Start the decompression
  jpeg_start_decompress(&cinfo);

  // Get the initial row pointers
  const unsigned num_row_ptrs = std::min(max_row_ptrs, numRecommendedRowPtrs());
  row_ptrs.resize(num_row_ptrs, NULL);
  fill(rows, num_row_ptrs, true);

  unsigned base = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    assert(cinfo.output_scanline >= base &&
           cinfo.output_scanline <= base + num_row_ptrs);

    // Get some new row pointers from the user
    if (cinfo.output_scanline == base + num_row_ptrs) {
      fill(rows, num_row_ptrs, false);
      base += num_row_ptrs;
    }
    assert(cinfo.output_scanline >= base &&
           cinfo.output_scanline < base + num_row_ptrs);

    const unsigned index = cinfo.output_scanline - base;
    assert(index >= 0 && index < num_row_ptrs);
    jpeg_read_scanlines(&cinfo, &row_ptrs[index], num_row_ptrs - index);
  }

  // Copy color map before jpeg_finish_decompress() nukes it.
  cmap.resize(cinfo.actual_number_of_colors * cinfo.out_color_components);
  for (int i = 0; i < cinfo.out_color_components; ++i)
    for (int j = 0; j < cinfo.actual_number_of_colors; ++j)
      cmap[j * cinfo.out_color_components + i] = cinfo.colormap[i][j];

  jpeg_finish_decompress(&cinfo);
}

#endif
