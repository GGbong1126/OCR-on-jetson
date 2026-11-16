#pragma once
#include <csetjmp>
#include <cstdint>
#include "NvJpegDecoder.h"

namespace jpeg_header {
struct Error { jpeg_error_mgr base; std::jmp_buf jump; };
inline void fail(j_common_ptr info) { std::longjmp(reinterpret_cast<Error*>(info->err)->jump,1); }
struct Header { std::uint32_t width,height; int status; };
inline Header inspect(unsigned char* bytes,unsigned long count,std::uint64_t max_pixels=32000000) {
    jpeg_decompress_struct info{}; Error error{};
    info.err=jpeg_std_error(&error.base); error.base.error_exit=fail;
    if (setjmp(error.jump)) { jpeg_destroy_decompress(&info); return {0,0,2}; }
    jpeg_create_decompress(&info); jpeg_mem_src(&info,bytes,count); jpeg_read_header(&info,TRUE);
    Header result{info.image_width,info.image_height,0};
    if (!result.width || !result.height || std::uint64_t(result.width)*result.height>max_pixels) result.status=2;
    else if (info.progressive_mode || info.data_precision!=8 || info.jpeg_color_space!=JCS_YCbCr || info.num_components!=3 ||
             info.comp_info[0].h_samp_factor!=2 || info.comp_info[0].v_samp_factor!=2 ||
             info.comp_info[1].h_samp_factor!=1 || info.comp_info[1].v_samp_factor!=1 ||
             info.comp_info[2].h_samp_factor!=1 || info.comp_info[2].v_samp_factor!=1) result.status=3;
    jpeg_destroy_decompress(&info); return result;
}
} // namespace jpeg_header
