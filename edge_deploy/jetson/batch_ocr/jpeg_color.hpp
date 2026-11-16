#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace jpeg_color {
// JPEG pixel-centre triangle interpolation; alternate rounding matches IJG h2v2.
inline int chroma(const unsigned char* plane, std::size_t stride, unsigned cw,
                  unsigned ch, unsigned x, unsigned y) {
    const unsigned cx=x/2, cy=y/2;
    if (cw<=2) return plane[std::size_t(cy)*stride+cx];
    const unsigned ny=(y%2) ? std::min(cy+1,ch-1) : (cy ? cy-1 : 0);
    const unsigned nx=(x%2) ? std::min(cx+1,cw-1) : (cx ? cx-1 : 0);
    const auto sum=[&](unsigned column) {
        return 3*int(plane[std::size_t(cy)*stride+column])+int(plane[std::size_t(ny)*stride+column]);
    };
    return (3*sum(cx)+sum(nx)+(x%2 ? 7 : 8))/16;
}

inline int floor16(int value) { return value>=0 ? value/65536 : -((-value+65535)/65536); }
inline unsigned char clip(int value) { return static_cast<unsigned char>(std::max(0,std::min(255,value))); }
inline std::array<unsigned char,3> rgb(int y,int cb,int cr) {
    const int u=cb-128,v=cr-128;
    return {{clip(y+floor16(91881*v+32768)),
             clip(y+floor16(-22554*u-46802*v+32768)),
             clip(y+floor16(116130*u+32768))}};
}
}  // namespace jpeg_color
