#pragma once
#include <string>
#include <vector>

#include <NvInfer.h>

#define TRT_10

typedef struct IOInfomation
{
    std::string name;
    nvinfer1::Dims dims;
    uint32_t total_bytes;   // 总共占用的字节数，= 1 * 3 * 640 * 640 * element_size
    void* d_ptr;
    void* h_ptr;

    IOInfomation() : 
        total_bytes(0), d_ptr(nullptr), h_ptr(nullptr), dims({}) {};
}IOInfo;

typedef struct BoundingBox
{
    int x, y;
    int width, height;
    int class_index;
    float conf_threshold;
};

class TensorRTInferenceEngine
{
public:
    
    TensorRTInferenceEngine();

    
    ~TensorRTInferenceEngine();

    bool Initialize(const std::string& enginePath);
    
    void Infer(const std::vector<void*>& d_inputPtrs);




private:
    
    void CopyInputsToCuda(const std::vector<void*>& h_ptrs = std::vector<void*>());

    void CopyOutputsToCpu(const std::vector<void*> & h_ptrs = std::vector<void*>());

private:
    void MakePipe();

    void InferInternal();
private:
    // Logger用于TensorRT输出调试信息，可根据需要扩展日志细节
    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    } m_Logger;

    nvinfer1::IRuntime* m_Runtime{ nullptr };
    nvinfer1::ICudaEngine* m_Engine{ nullptr };
    nvinfer1::IExecutionContext* m_Context{ nullptr };
    cudaStream_t m_Stream{ nullptr };
    
    std::vector<IOInfo> m_inputs;
    std::vector<IOInfo> m_outputs;
};
