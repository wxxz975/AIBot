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
    // 构造函数：加载序列化引擎并初始化执行上下文和CUDA流
    TensorRTInferenceEngine();

    // 析构函数：释放TensorRT资源与CUDA流
    ~TensorRTInferenceEngine();

    bool Initialize(const std::string& enginePath);

    // 推理接口（适用于单个输入、单个输出的简单场景）
    // inputCudaPtr：设备端输入数据指针
    // outputCudaPtr：设备端输出缓冲区指针（需要预先分配好足够空间）
    // batchSize：推理的批量大小
    // inputName和outputName允许根据不同模型进行灵活绑定
    void Infer(void* inputCudaPtr);

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
