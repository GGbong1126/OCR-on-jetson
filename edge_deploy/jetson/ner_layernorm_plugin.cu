// E116 isolated BERT LayerNorm: FP32 only, [1,S,768], epsilon from source.
// Two-pass mean/variance matches the ONNX operation order (not reduction order).
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
using namespace nvinfer1;
namespace autoocr_ner_norm {
constexpr char kType[] = "AutoOCRNerLayerNormFP32";
constexpr uint32_t kMagic = 0x4e4c3031;
constexpr int kWidth = 768, kThreads = 256;

__device__ float reduce(float value, float* scratch) {
    int t = threadIdx.x;
    scratch[t] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride /= 2) {
        if (t < stride) scratch[t] = __fadd_rn(scratch[t], scratch[t + stride]);
        __syncthreads();
    }
    return scratch[0];
}
__global__ void layernorm(const float* x, const float* gamma, const float* beta,
                          float* y, float eps) {
    __shared__ float scratch[kThreads];
    const int row = blockIdx.x * kWidth, t = threadIdx.x;
    float sum = 0.0f;
    for (int j = t; j < kWidth; j += kThreads) sum = __fadd_rn(sum, x[row+j]);
    float mean = __fdiv_rn(reduce(sum, scratch), float(kWidth));
    // All lanes must finish reading scratch[0] before the next reduction writes.
    __syncthreads();
    float squares = 0.0f;
    for (int j = t; j < kWidth; j += kThreads) {
        float d = __fsub_rn(x[row+j], mean);
        squares = __fadd_rn(squares, __fmul_rn(d, d));
    }
    float var = __fdiv_rn(reduce(squares, scratch), float(kWidth));
    float denominator = __fsqrt_rn(__fadd_rn(var, eps));
    for (int j = t; j < kWidth; j += kThreads) {
        float norm = __fdiv_rn(__fsub_rn(x[row+j], mean), denominator);
        y[row+j] = __fadd_rn(__fmul_rn(norm, gamma[j]), beta[j]);
    }
}

class Norm final : public IPluginV2DynamicExt {
    float epsilon;
    std::string ns;
public:
    explicit Norm(float eps) : epsilon(eps) {}
    const char* getPluginType() const noexcept override { return kType; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* s) noexcept override { ns = s ? s : ""; }
    const char* getPluginNamespace() const noexcept override { return ns.c_str(); }
    IPluginV2DynamicExt* clone() const noexcept override { return new(std::nothrow) Norm(*this); }
    size_t getSerializationSize() const noexcept override { return sizeof(kMagic)+sizeof(epsilon); }
    void serialize(void* p) const noexcept override {
        std::memcpy(p, &kMagic, sizeof(kMagic));
        std::memcpy(static_cast<char*>(p)+sizeof(kMagic), &epsilon, sizeof(epsilon));
    }
    DataType getOutputDataType(int, const DataType*, int) const noexcept override { return DataType::kFLOAT; }
    DimsExprs getOutputDimensions(int, const DimsExprs* in, int n, IExprBuilder&) noexcept override {
        if (n == 3 && in[0].nbDims == 3 && in[1].nbDims == 1 && in[2].nbDims == 1) return in[0];
        return DimsExprs{};
    }
    bool supportsFormatCombination(int pos, const PluginTensorDesc* io, int ni, int no) noexcept override {
        return ni==3 && no==1 && pos>=0 && pos<4 && io[pos].type==DataType::kFLOAT && io[pos].format==TensorFormat::kLINEAR;
    }
    void configurePlugin(const DynamicPluginTensorDesc*, int, const DynamicPluginTensorDesc*, int) noexcept override {}
    size_t getWorkspaceSize(const PluginTensorDesc*, int, const PluginTensorDesc*, int) const noexcept override { return 0; }
    int enqueue(const PluginTensorDesc* in, const PluginTensorDesc* out,
                const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (!in || !out || !inputs || !outputs) return 1;
        const auto& x = in[0].dims;
        if (x.nbDims!=3 || x.d[0]!=1 || x.d[1]<2 || x.d[1]>512 || x.d[2]!=kWidth) return 1;
        for (int i=0; i<3; ++i) {
            if (!inputs[i] || in[i].type!=DataType::kFLOAT || in[i].format!=TensorFormat::kLINEAR) return 1;
            if (i && (in[i].dims.nbDims!=1 || in[i].dims.d[0]!=kWidth)) return 1;
        }
        if (!outputs[0] || out[0].type!=DataType::kFLOAT || out[0].format!=TensorFormat::kLINEAR || out[0].dims.nbDims!=3) return 1;
        for (int i=0; i<3; ++i) if (out[0].dims.d[i]!=x.d[i]) return 1;
        layernorm<<<x.d[1],kThreads,0,stream>>>(static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),static_cast<const float*>(inputs[2]),static_cast<float*>(outputs[0]),epsilon);
        return cudaPeekAtLastError()==cudaSuccess ? 0 : 1;
    }
};
class Creator final : public IPluginCreator {
    std::string ns;
    PluginField field{"epsilon",nullptr,PluginFieldType::kFLOAT32,1};
    PluginFieldCollection fields{1,&field};
public:
    const char* getPluginName() const noexcept override { return kType; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &fields; }
    void setPluginNamespace(const char* s) noexcept override { ns=s ? s : ""; }
    const char* getPluginNamespace() const noexcept override { return ns.c_str(); }
    IPluginV2* make(float eps) noexcept {
        if (!std::isfinite(eps) || eps<=0 || eps>1) return nullptr;
        auto* p=new(std::nothrow) Norm(eps); if (p) p->setPluginNamespace(ns.c_str()); return p;
    }
    IPluginV2* createPlugin(const char*, const PluginFieldCollection* f) noexcept override {
        if (!f || f->nbFields!=1 || !f->fields || !f->fields[0].name || std::strcmp(f->fields[0].name,"epsilon") ||
            f->fields[0].type!=PluginFieldType::kFLOAT32 || f->fields[0].length!=1 || !f->fields[0].data) return nullptr;
        float eps; std::memcpy(&eps,f->fields[0].data,sizeof(eps)); return make(eps);
    }
    IPluginV2* deserializePlugin(const char*, const void* data, size_t size) noexcept override {
        uint32_t magic; float eps;
        if (!data || size!=sizeof(magic)+sizeof(eps)) return nullptr;
        std::memcpy(&magic,data,sizeof(magic));
        std::memcpy(&eps,static_cast<const char*>(data)+sizeof(magic),sizeof(eps));
        return magic==kMagic ? make(eps) : nullptr;
    }
};
REGISTER_TENSORRT_PLUGIN(Creator);
}
