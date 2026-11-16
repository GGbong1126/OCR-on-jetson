#pragma once
#include "jpeg_header.hpp"
#include "jpeg_color.hpp"
#include <chrono>
#include <memory>
#include <vector>

namespace autoocr {
struct JpegPixels {
    int status=3;
    std::uint32_t width=0,height=0;
    std::vector<unsigned char> bgr;
    double decode_ms=0,color_ms=0;
};
class JpegDecoder {
public:
    JpegPixels decode(const unsigned char* bytes,std::size_t size) {
        JpegPixels result;
        if (size<4 || bytes[0]!=255 || bytes[1]!=216) return result;
        auto h=jpeg_header::inspect(const_cast<unsigned char*>(bytes),size);
        result.status=h.status;
        if (h.status) return result;
        if (h.width>8192 || h.height>8192) { result.status=2; return result; }
        // shortcut: raw <=24MB keeps two-copy Python transport plus a result within 64MiB.
        if (h.width%2 || h.height%2 || std::uint64_t(h.width)*h.height>8000000 ||
            bytes[size-2]!=255 || bytes[size-1]!=217) { result.status=3; return result; }
        auto start=std::chrono::steady_clock::now();
        std::unique_ptr<NvJPEGDecoder> decoder(NvJPEGDecoder::createJPEGDecoder("ocr-input"));
        if (!decoder) { result.status=2; return result; }
        NvBuffer* pointer=nullptr; std::uint32_t format=0,w=0,height=0;
        const int code=decoder->decodeToBuffer(&pointer,const_cast<unsigned char*>(bytes),size,&format,&w,&height);
        auto color_start=std::chrono::steady_clock::now();
        result.decode_ms=std::chrono::duration<double,std::milli>(color_start-start).count();
        std::unique_ptr<NvBuffer> buffer(pointer);
        if (code || decoder->isInError() || !buffer || w!=h.width || height!=h.height ||
            format!=V4L2_PIX_FMT_YUV420M || buffer->n_planes!=3) { result.status=2; return result; }
        for (unsigned i=0;i<3;++i) {
            const auto& p=buffer->planes[i];
            if (!p.data || p.fmt.width!=w/(i?2:1) || p.fmt.height!=height/(i?2:1) ||
                p.fmt.bytesperpixel!=1 || p.mem_offset || p.fmt.stride<p.fmt.width) { result.status=2; return result; }
        }
        result.width=w; result.height=height; result.bgr.resize(std::size_t(w)*height*3);
        for (unsigned y=0;y<height;++y) for (unsigned x=0;x<w;++x) {
            auto sample=[&](unsigned i) { const auto& p=buffer->planes[i]; return jpeg_color::chroma(p.data,p.fmt.stride,w/2,height/2,x,y); };
            auto rgb=jpeg_color::rgb(buffer->planes[0].data[std::size_t(y)*buffer->planes[0].fmt.stride+x],sample(1),sample(2));
            const auto offset=(std::size_t(y)*w+x)*3;
            result.bgr[offset]=rgb[2]; result.bgr[offset+1]=rgb[1]; result.bgr[offset+2]=rgb[0];
        }
        result.color_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-color_start).count();
        result.status=0; return result;
    }
};
} // namespace autoocr
