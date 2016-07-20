#ifndef AJS_JPEG_H
#define AJS_JPEG_H

extern "C" {
    #include <jpeglib.h>
}

/// Collection of libjpeg-related constants.
namespace JPEG {

    /// Color space conversions.
    typedef enum {
        COLOR_UNKNOWN   = JCS_UNKNOWN,      ///< Unknown or not specified
        COLOR_GRAYSCALE = JCS_GRAYSCALE,    ///< Monochrome 
        COLOR_RGB       = JCS_RGB,          ///< Red/green/blue 
        COLOR_YCC       = JCS_YCbCr,        ///< Y/Cb/Cr (also known as YUV) 
        COLOR_CMYK      = JCS_CMYK,         ///< C/M/Y/K
        COLOR_YCCK      = JCS_YCCK          ///< Y/Cb/Cr/K 
    } ColorSpace;
    
    /// Extraction scale factors.
    typedef enum {
        SCALE_FULL_SIZE = 1,                ///< Extract the full resolution
        SCALE_HALF      = 2,                ///< Extract at half-resolution
        SCALE_QUARTER   = 4,                ///< Extract at quarter-resolution
        SCALE_EIGHTH    = 8                 ///< Extract at eighth-resolution
    } Scale;

    /// Dither modes for quantization.
    typedef enum {
        DITHER_NONE    = JDITHER_NONE,      ///< No dithering, fast and ugly
        DITHER_ORDERED = JDITHER_ORDERED,   ///< Ordered dithering, moderate in both
        DITHER_FS      = JDITHER_FS         ///< Floyd-Steinberg dithering, flow and pretty
    } Dither;
    
    /// Time/quality tradeoff.
    typedef enum {
        FASTER,                             ///< Try to go faster at the expense of quality
        DEFAULT,                            ///< Use libjpeg's defaults
        BETTER                              ///< Try to generate higher quality at the expense of speed
    } TimeQualityTradeoff;
    
    
}

#endif
