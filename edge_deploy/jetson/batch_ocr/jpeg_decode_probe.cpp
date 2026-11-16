// Isolated M1a diagnostic, not a production decoder or a performance benchmark.
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <csetjmp>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <unistd.h>

#include "NvJpegDecoder.h"
#include "jpeg_color.hpp"
#include "jpeg_header.hpp"

namespace {
constexpr std::size_t max_bytes = 16 * 1024 * 1024;

int reject(int code, const char* reason) {
    std::cout << "{\"status\":\"rejected\",\"reason\":\"" << reason << "\"}\n";
    return code;
}

bool save_rgb(const NvBuffer& buffer, std::uint32_t width, std::uint32_t height, const char* path) {
    if (width % 2 || height % 2) return false;
    for (unsigned i = 0; i < 3; ++i) {
        const auto& p = buffer.planes[i];
        if (p.fmt.width != width / (i ? 2 : 1) || p.fmt.height != height / (i ? 2 : 1) ||
            p.fmt.bytesperpixel != 1 || p.mem_offset != 0) return false;
    }
    std::vector<unsigned char> row(std::size_t(width) * 3);
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return false;
    bool ok = true;
    for (std::uint32_t y = 0; y < height && ok; ++y) {
        const auto* yy = buffer.planes[0].data + std::size_t(y) * buffer.planes[0].fmt.stride;
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto sample=[&](unsigned i) {
                return jpeg_color::chroma(buffer.planes[i].data,buffer.planes[i].fmt.stride,width/2,height/2,x,y);
            };
            const auto color=jpeg_color::rgb(yy[x],sample(1),sample(2));
            std::copy(color.begin(),color.end(),row.begin()+std::size_t(x)*3);
        }
        std::size_t offset = 0;
        while (offset < row.size()) {
            const auto count = write(fd, row.data() + offset, row.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) { ok = false; break; }
            offset += std::size_t(count);
        }
    }
    if (close(fd)) ok = false;
    if (!ok) unlink(path);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return reject(2, "input_and_optional_rgb_output_required");
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) return reject(2, "input_unreadable");
    const auto size = file.tellg();
    if (size < 4 || size > static_cast<std::streamoff>(max_bytes))
        return reject(2, "input_byte_limit");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size))
        return reject(2, "input_read_failed");
    if (bytes[0] != 0xff || bytes[1] != 0xd8)
        return reject(3, "not_jpeg_probe_input");
    const auto expected = jpeg_header::inspect(bytes.data(), bytes.size());
    if (expected.status == 2) return reject(2, "invalid_header_or_pixel_limit");
    if (expected.status == 3) return reject(3, "outside_baseline_ycbcr420_probe_scope");
    if (argc == 3 && (expected.width % 2 || expected.height % 2))
        return reject(3, "odd_size_requires_software_until_validated");

    std::unique_ptr<NvJPEGDecoder> decoder(NvJPEGDecoder::createJPEGDecoder("ocr-jpeg-probe"));
    if (!decoder) return reject(2, "decoder_creation_failed");
    NvBuffer* raw = nullptr;
    std::uint32_t format = 0, width = 0, height = 0;
    const auto start = std::chrono::steady_clock::now();
    const int status = decoder->decodeToBuffer(&raw, bytes.data(), bytes.size(), &format, &width, &height);
    const auto finish = std::chrono::steady_clock::now();
    std::unique_ptr<NvBuffer> buffer(raw);
    if (status || decoder->isInError() || !buffer)
        return reject(2, "accelerated_decode_failed");
    if (width != expected.width || height != expected.height || buffer->n_planes != 3)
        return reject(2, "decoded_geometry_mismatch");
    for (std::uint32_t i = 0; i < buffer->n_planes; ++i) {
        const auto& p = buffer->planes[i];
        if (!p.data || !p.fmt.width || !p.fmt.height || !p.fmt.bytesperpixel ||
            p.fmt.stride < std::uint64_t(p.fmt.width) * p.fmt.bytesperpixel)
            return reject(2, "invalid_plane_layout");
    }
    const auto color_start = std::chrono::steady_clock::now();
    if (argc == 3 && (format != V4L2_PIX_FMT_YUV420M || !save_rgb(*buffer, width, height, argv[2])))
        return reject(2, "rgb_layout_or_exclusive_write_failed");
    const double color_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - color_start).count();
    const double ms = std::chrono::duration<double, std::milli>(finish - start).count();
    std::cout << "{\"status\":\"decoded_geometry_checked_not_ocr_validated\","
              << "\"width\":" << width << ",\"height\":" << height
              << ",\"pixel_format\":" << format << ",\"decode_to_buffer_ms\":" << ms
              << ",\"planes\":[";
    for (std::uint32_t i = 0; i < buffer->n_planes; ++i) {
        const auto& p = buffer->planes[i].fmt;
        if (i) std::cout << ',';
        std::cout << "{\"width\":" << p.width << ",\"height\":" << p.height
                  << ",\"bytes_per_pixel\":" << p.bytesperpixel << ",\"stride\":" << p.stride << '}';
    }
    std::cout << ']';
    if (argc == 3)
        std::cout << ",\"rgb_format\":\"RGB24_fancy_full_range_candidate\",\"rgb_written_bytes\":"
                  << std::uint64_t(width) * height * 3 << ",\"color_convert_and_write_ms\":" << color_ms;
    std::cout << "}\n";
    return 0;
}
