#include "jpeg_color.hpp"
#include <cassert>
#include <iostream>

int main() {
    const unsigned char p[]={0,40,80,255,120,160,200,255};
    const int expected[4][6]={{0,10,30,50,70,80},{30,40,60,80,100,110},
                            {90,100,120,140,160,170},{120,130,150,170,190,200}};
    for (unsigned y=0;y<4;++y) for (unsigned x=0;x<6;++x)
        assert(jpeg_color::chroma(p,4,3,2,x,y)==expected[y][x]);
    const unsigned char small[]={3,200};
    for (unsigned x=0;x<4;++x) assert(jpeg_color::chroma(small,2,2,1,x,0)==small[x/2]);
    for (int y=0;y<256;++y) {
        const auto c=jpeg_color::rgb(y,128,128);
        assert(c[0]==y && c[1]==y && c[2]==y);
    }
    assert((jpeg_color::rgb(76,85,255)==std::array<unsigned char,3>{{254,0,0}}));
    assert((jpeg_color::rgb(29,255,107)==std::array<unsigned char,3>{{0,0,254}}));
    assert(jpeg_color::floor16(-1)==-1 && jpeg_color::floor16(-65536)==-1);
    std::cout<<"JPEG colour edge/stride/rounding tests passed\n";
}
