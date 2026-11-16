// AutoOCR: narrowly scoped ONNX GridSample-16, FP32 NCHW, bilinear/zeros,
// align_corners=false. No dependence on PyTorch, cuDNN, or host callbacks.
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>

using namespace nvinfer1;
namespace autoocr_grid {
constexpr uint32_t kMagic = 0x47533031;
constexpr char kType[] = "AutoOCRGridSample";

__device__ float pixel(const float* x, int y, int c, int h, int w) {
    return y >= 0 && y < h && c >= 0 && c < w ? x[y * w + c] : 0.0f;
}
__global__ void sample(const float* x, const float* grid, float* out,
                       int count, int channels, int h, int w, int oh, int ow) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const int pos = i % (oh * ow), nc = i / (oh * ow);
        const int g = ((nc / channels) * oh * ow + pos) * 2;
        float gx = grid[g], gy = grid[g + 1];
        if (isnan(gx)) gx = -1.0f;
        if (isnan(gy)) gy = -1.0f;
        // Match the fused multiply/subtract used by the board's ARM ORT build.
        float px = fmaf(gx + 1.0f, float(w), -1.0f) * 0.5f;
        float py = fmaf(gy + 1.0f, float(h), -1.0f) * 0.5f;
        // Reject outside the one-pixel interpolation fringe BEFORE int cast.
        if (!(px > -1 && px < w && py > -1 && py < h)) { out[i] = 0; continue; }
        int x0 = static_cast<int>(floorf(px)), y0 = static_cast<int>(floorf(py));
        float dx = px - x0, dy = py - y0;
        const float* p = x + nc * h * w;
        out[i] = pixel(p, y0, x0, h, w) * (1-dx) * (1-dy)
               + pixel(p, y0, x0+1, h, w) * dx * (1-dy)
               + pixel(p, y0+1, x0, h, w) * (1-dx) * dy
               + pixel(p, y0+1, x0+1, h, w) * dx * dy;
    }
}

class Grid final : public IPluginV2DynamicExt {
    std::string ns;
public:
    const char* getPluginType() const noexcept override { return kType; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* s) noexcept override { ns = s ? s : ""; }
    const char* getPluginNamespace() const noexcept override { return ns.c_str(); }
    IPluginV2DynamicExt* clone() const noexcept override { return new(std::nothrow) Grid(*this); }
    size_t getSerializationSize() const noexcept override { return sizeof(kMagic); }
    void serialize(void* p) const noexcept override { std::memcpy(p, &kMagic, sizeof(kMagic)); }
    DataType getOutputDataType(int, const DataType*, int) const noexcept override { return DataType::kFLOAT; }
    DimsExprs getOutputDimensions(int, const DimsExprs* in, int n, IExprBuilder&) noexcept override {
        DimsExprs d{};
        if (n != 2 || in[0].nbDims != 4 || in[1].nbDims != 4) return d;
        d.nbDims = 4; d.d[0] = in[0].d[0]; d.d[1] = in[0].d[1];
        d.d[2] = in[1].d[1]; d.d[3] = in[1].d[2]; return d;
    }
    bool supportsFormatCombination(int pos, const PluginTensorDesc* io, int ni, int no) noexcept override {
        return ni == 2 && no == 1 && pos >= 0 && pos < 3
            && io[pos].type == DataType::kFLOAT && io[pos].format == TensorFormat::kLINEAR;
    }
    void configurePlugin(const DynamicPluginTensorDesc*, int, const DynamicPluginTensorDesc*, int) noexcept override {}
    size_t getWorkspaceSize(const PluginTensorDesc*, int, const PluginTensorDesc*, int) const noexcept override { return 0; }
    int enqueue(const PluginTensorDesc* in, const PluginTensorDesc*,
                const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        const Dims& x = in[0].dims; const Dims& g = in[1].dims;
        if (x.nbDims != 4 || g.nbDims != 4 || x.d[0] != g.d[0] || g.d[3] != 2) return 1;
        int64_t count = 1, volume = 1, grids = 1;
        for (int i=0; i<4; ++i) {
            if (x.d[i] <= 0 || g.d[i] <= 0) return 1;
            volume *= x.d[i]; grids *= g.d[i];
        }
        count = int64_t(x.d[0])*x.d[1]*g.d[1]*g.d[2];
        if (count > INT32_MAX || volume > INT32_MAX || grids > INT32_MAX) return 1;
        int blocks = static_cast<int>((count+255)/256); if (blocks > 4096) blocks=4096;
        sample<<<blocks,256,0,stream>>>(static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]), static_cast<float*>(outputs[0]),
            count, x.d[1], x.d[2], x.d[3], g.d[1], g.d[2]);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }
};
class Creator final : public IPluginCreator {
    std::string ns;
    PluginFieldCollection fields{0, nullptr};
public:
    const char* getPluginName() const noexcept override { return kType; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &fields; }
    void setPluginNamespace(const char* s) noexcept override { ns = s ? s : ""; }
    const char* getPluginNamespace() const noexcept override { return ns.c_str(); }
    IPluginV2* createPlugin(const char*, const PluginFieldCollection* f) noexcept override {
        if (f && f->nbFields != 0) return nullptr;
        auto* p = new(std::nothrow) Grid; if (p) p->setPluginNamespace(ns.c_str()); return p;
    }
    IPluginV2* deserializePlugin(const char*, const void* data, size_t size) noexcept override {
        uint32_t v = 0; if (!data || size != sizeof(v)) return nullptr;
        std::memcpy(&v, data, size); if (v != kMagic) return nullptr;
        return createPlugin(nullptr, nullptr);
    }
};
REGISTER_TENSORRT_PLUGIN(Creator);
}
