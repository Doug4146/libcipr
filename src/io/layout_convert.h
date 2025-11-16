/**
 * @file layout_convert.h
 *
 * Internal header for the pixel layout conversion (including stride-based
 * padding) operations.
 */

#ifndef LIBCIPR_IO_LAYOUT_CONVERT_H_
#define LIBCIPR_IO_LAYOUT_CONVERT_H_

#include "utils/utils.h"

/**
 * @brief Converts a non-padded interleaved RGB8 buffer to stride-padded planar
 * layout.
 *
 * Takes an input buffer in interleaved RGB8 format (RGBRGBRGB..) without any
 * row padding and converts it into planar layout (RRRGGGBBB..) with each row
 * of each planar region being padded to a multiple of `stride`.
 *
 * @param dst_buffer Pointer to the padded planar RGB8 output buffer.
 * @param src_buffer Pointer to the non-padded interleaved RGB8 input buffer.
 */
void cipr__layout_interleaved_to_planar_RGB8(cipr_u8 *dst_buffer, cipr_u8 *src_buffer, cipr_i32 h,
                                             cipr_i32 w, cipr_i32 stride);

/**
 * @brief Converts a stride-padded planar RGB8 buffer to non-padded interleaved
 * layout.
 *
 * Takes an input buffer in planar RGB8 format (RRR...GGG...BBB...) where each
 * row of each plane is padded to a multiple of `stride`, and converts it into
 * tightly packed interleaved layout (RGBRGBRGB...) without any row padding.
 *
 * @param dst_buffer Pointer to the non-padded interleaved RGB8 output buffer.
 * @param src_buffer Pointer to the padded planar RGB8 input buffer.
 */
void cipr__layout_planar_to_interleaved_RGB8(cipr_u8 *dst_buffer, cipr_u8 *src_buffer, cipr_i32 h,
                                             cipr_i32 w, cipr_i32 stride);

/**
 * @brief Removes row padding from a stride-padded GRAY8 buffer.
 *
 * Takes an input buffer in GRAY8 format where each row is padded to a multiple
 * of `stride` and converts it into tightly packed layout with no row padding.
 *
 * @param dst_buffer Pointer to the non-padded GRAY8 output buffer.
 * @param src_buffer Pointer to the padded GRAY8 input buffer.
 */
void cipr__layout_unpad_GRAY8(cipr_u8 *dst_buffer, cipr_u8 *src_buffer, cipr_i32 h, cipr_i32 w,
                              cipr_i32 stride);

#endif // LIBCIPR_IO_LAYOUT_CONVERT_H_