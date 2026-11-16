// E118: original BERT score/8 + (1-mask)*(-FLT_MAX), then FP32 Softmax.
// Finite-mask rounding is intentional, including uniform all-masked rows.
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
using namespace nvinfer1;
namespace autoocr_ner_mask_softmax {
constexpr char kType[]="AutoOCRNerMaskSoftmaxFP32";
constexpr uint32_t kMagic=0x4d533031;
constexpr int kThreads=256;
__global__ void softmax(const float* scores,const float* mask,float* y,int width) {
    __shared__ float buf[kThreads];
    int t=threadIdx.x,row=blockIdx.x*width;
    float values[2],maximum=-FLT_MAX;
    for(int i=0;i<2;++i) {
        int j=t+i*kThreads;
        values[i]=j<width ? __fadd_rn(__fdiv_rn(scores[row+j],8.0f),
            __fmul_rn(__fsub_rn(1.0f,mask[j]),-FLT_MAX)) : -FLT_MAX;
        if(j<width) maximum=fmaxf(maximum,values[i]);
    }
    buf[t]=maximum;__syncthreads();
    for(int step=kThreads/2;step;step/=2) {
        if(t<step)buf[t]=fmaxf(buf[t],buf[t+step]);
        __syncthreads();
    }
    maximum=buf[0];__syncthreads();
    float sum=0.0f;
    for(int i=0;i<2;++i) {
        int j=t+i*kThreads;
        values[i]=j<width ? expf(__fsub_rn(values[i],maximum)) : 0.0f;
        sum=__fadd_rn(sum,values[i]);
    }
    buf[t]=sum;__syncthreads();
    for(int step=kThreads/2;step;step/=2) {
        if(t<step)buf[t]=__fadd_rn(buf[t],buf[t+step]);
        __syncthreads();
    }
    float denominator=buf[0];
    for(int i=0;i<2;++i) {
        int j=t+i*kThreads;
        if(j<width)y[row+j]=__fdiv_rn(values[i],denominator);
    }
}
class Softmax final:public IPluginV2DynamicExt {
    std::string ns;
public:
    const char* getPluginType()const noexcept override{return kType;}
    const char* getPluginVersion()const noexcept override{return "1";}
    int getNbOutputs()const noexcept override{return 1;}
    int initialize()noexcept override{return 0;}
    void terminate()noexcept override{}
    void destroy()noexcept override{delete this;}
    void setPluginNamespace(const char* s)noexcept override{ns=s?s:"";}
    const char* getPluginNamespace()const noexcept override{return ns.c_str();}
    IPluginV2DynamicExt* clone()const noexcept override{return new(std::nothrow) Softmax(*this);}
    size_t getSerializationSize()const noexcept override{return sizeof(kMagic);}
    void serialize(void* p)const noexcept override{std::memcpy(p,&kMagic,sizeof(kMagic));}
    DataType getOutputDataType(int,const DataType*,int)const noexcept override{return DataType::kFLOAT;}
    DimsExprs getOutputDimensions(int,const DimsExprs* in,int n,IExprBuilder&)noexcept override{
        if(n==2&&in[0].nbDims==4&&in[1].nbDims==2)return in[0];return DimsExprs{};
    }
    bool supportsFormatCombination(int pos,const PluginTensorDesc* io,int ni,int no)noexcept override{
        return ni==2&&no==1&&pos>=0&&pos<3&&io[pos].type==DataType::kFLOAT&&io[pos].format==TensorFormat::kLINEAR;
    }
    void configurePlugin(const DynamicPluginTensorDesc*,int,const DynamicPluginTensorDesc*,int)noexcept override{}
    size_t getWorkspaceSize(const PluginTensorDesc*,int,const PluginTensorDesc*,int)const noexcept override{return 0;}
    int enqueue(const PluginTensorDesc* in,const PluginTensorDesc* out,const void* const* inputs,
                void* const* outputs,void*,cudaStream_t stream)noexcept override{
        if(!in||!out||!inputs||!outputs||!inputs[0]||!inputs[1]||!outputs[0])return 1;
        auto x=in[0].dims;auto m=in[1].dims;
        if(x.nbDims!=4||x.d[0]!=1||x.d[1]!=12||x.d[2]<2||x.d[2]>512||x.d[3]!=x.d[2]||
           m.nbDims!=2||m.d[0]!=1||m.d[1]!=x.d[3]||out[0].dims.nbDims!=4)return 1;
        for(int i=0;i<2;++i)if(in[i].type!=DataType::kFLOAT||in[i].format!=TensorFormat::kLINEAR)return 1;
        if(out[0].type!=DataType::kFLOAT||out[0].format!=TensorFormat::kLINEAR)return 1;
        for(int i=0;i<4;++i)if(out[0].dims.d[i]!=x.d[i])return 1;
        softmax<<<12*x.d[2],kThreads,0,stream>>>(static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),static_cast<float*>(outputs[0]),x.d[3]);
        return cudaPeekAtLastError()==cudaSuccess?0:1;
    }
};
class Creator final:public IPluginCreator {
    std::string ns;
    PluginFieldCollection fields{0,nullptr};
public:
    const char* getPluginName()const noexcept override{return kType;}
    const char* getPluginVersion()const noexcept override{return "1";}
    const PluginFieldCollection* getFieldNames()noexcept override{return &fields;}
    void setPluginNamespace(const char* s)noexcept override{ns=s?s:"";}
    const char* getPluginNamespace()const noexcept override{return ns.c_str();}
    IPluginV2* make()noexcept{auto* p=new(std::nothrow) Softmax();if(p)p->setPluginNamespace(ns.c_str());return p;}
    IPluginV2* createPlugin(const char*,const PluginFieldCollection* f)noexcept override{
        return f&&f->nbFields==0?make():nullptr;
    }
    IPluginV2* deserializePlugin(const char*,const void* p,size_t size)noexcept override{
        uint32_t magic;if(!p||size!=sizeof(magic))return nullptr;std::memcpy(&magic,p,sizeof(magic));return magic==kMagic?make():nullptr;
    }
};
REGISTER_TENSORRT_PLUGIN(Creator);
}
