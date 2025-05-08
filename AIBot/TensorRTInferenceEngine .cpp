#include "TensorRTInferenceEngine .h"

#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "Filesystem.h"
#include "CudaDebug.hpp"


inline static int GetElementSize(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}


TensorRTInferenceEngine::TensorRTInferenceEngine()
{
    
}


TensorRTInferenceEngine::~TensorRTInferenceEngine()
{
    if (m_Stream)
        cudaStreamDestroy(m_Stream);
}

bool TensorRTInferenceEngine::Initialize(const std::string& enginePath)
{
    std::vector<std::uint8_t> engineData;
    if (!IFilesystem::ReadFile(enginePath, engineData)) {
        std::cout << "Failed to read the engine file, please check:" << enginePath << "\n";
        return false;
    }

    initLibNvInferPlugins(&this->m_Logger, "");

    m_Runtime = nvinfer1::createInferRuntime(this->m_Logger);
    if (m_Runtime == nullptr) {
        std::cout << "Failed to create infer runtime!\n";
        return false;
    }

    m_Engine = m_Runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (m_Engine == nullptr) {
        std::cout << "Failed to deserialize the engine data!\n";
        return false;
    }

    m_Context = m_Engine->createExecutionContext();
    if (m_Context == nullptr) {
        std::cout << "Failed to createExecutionContext!\n";
        return false;
    }

    cudaStreamCreate(&m_Stream);

    int numIOPort = 0;
#ifdef TRT_10
    numIOPort = this->m_Engine->getNbIOTensors();
#else
    m_numIOPort = m_numIOPort = m_Engine->getNbBindings();
#endif

    for (int i = 0; i < numIOPort; ++i) {
        IOInfo          binding;
        nvinfer1::Dims  dims;

#ifdef TRT_10
        std::string        name = m_Engine->getIOTensorName(i);
        nvinfer1::DataType dtype = m_Engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = m_Engine->getBindingDataType(i);
        std::string        name = m_Engine->getBindingName(i);
#endif
        binding.name = name;
        
#ifdef TRT_10
        bool IsInput = m_Engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = engine->bindingIsInput(i);
#endif
        if (IsInput) {

#ifdef TRT_10
            dims = m_Engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            m_Context->setInputShape(name.c_str(), dims);
#else
            dims = m_Engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            m_Context->setBindingDimensions(i, dims);
#endif
            binding.dims = dims;
            binding.total_bytes = std::accumulate(dims.d, dims.d + dims.nbDims, GetElementSize(dtype), std::multiplies<int64_t>());
            m_inputs.emplace_back(binding);
        }
        else {
#ifdef TRT_10
            dims = m_Context->getTensorShape(name.c_str());
#else
            dims = m_Context->getBindingDimensions(i);
#endif
            //binding.size = get_size_by_dims(dims);

            binding.dims = dims;
            binding.total_bytes = std::accumulate(dims.d, dims.d + dims.nbDims, GetElementSize(dtype), std::multiplies<int64_t>());
            m_outputs.emplace_back(binding);
        }
    }

    for (int i = 0; i < 10; i++) {
        for (auto& bindings : m_inputs) {
            size_t size = bindings.total_bytes;
            void* h_ptr = malloc(size);
            memset(h_ptr, 0, size);
            CUDA_CHECK(cudaMemcpyAsync(bindings.d_ptr, h_ptr, size, cudaMemcpyHostToDevice, m_Stream));
            free(h_ptr);
        }
        InferInternal();
    }
}


void TensorRTInferenceEngine::Infer(const std::vector<void*>& d_inputPtrs)
{
    assert(m_inputs.size() == d_inputPtrs.size()); // only support one dims
    
    for (int idx = 0; idx < m_inputs.size(); ++idx) {
        size_t isize = m_inputs[idx].total_bytes;
        CUDA_CHECK(cudaMemcpy(m_inputs[idx].d_ptr, d_inputPtrs[idx], isize, cudaMemcpyDeviceToDevice));
    }

    InferInternal();

    // copy data from gpu memory to memeory host 
    for (auto& bindings : m_outputs) {
        size_t osize = bindings.total_bytes;
        CUDA_CHECK(cudaMemcpy(
            bindings.h_ptr, bindings.d_ptr, osize, cudaMemcpyDeviceToHost));
    }
}

void TensorRTInferenceEngine::CopyInputsToCuda(const std::vector<void*>& h_ptrs)
{
    std::vector<void*> d_temp_ptrs;
    bool customInput = false;
    if (h_ptrs.size() != m_inputs.size()) customInput = true;

    for (int idx = 0; idx < m_inputs.size(); ++idx) {
        cudaMemcpy(m_inputs[idx].d_ptr, customInput ? h_ptrs[idx] : m_inputs[idx].h_ptr, m_inputs[idx].total_bytes, cudaMemcpyHostToDevice);
    }
}

void TensorRTInferenceEngine::CopyOutputsToCpu(const std::vector<void*>& h_ptrs)
{
    bool customOutput = false;
    if (h_ptrs.size() != m_outputs.size()) customOutput = true;

    for (int idx = 0; idx < m_outputs.size(); ++idx) {
        cudaMemcpy(m_outputs[idx].d_ptr, customOutput ? h_ptrs[idx] : m_outputs[idx].d_ptr, m_outputs[idx].total_bytes, cudaMemcpyDeviceToHost);
    }
}

void TensorRTInferenceEngine::MakePipe()
{
    for (auto& bindings : m_inputs) {
        void* d_ptr;
        CUDA_CHECK(cudaMallocAsync(&d_ptr, bindings.total_bytes, m_Stream));
        bindings.d_ptr = d_ptr;
        
#ifdef TRT_10
        auto name = bindings.name.c_str();
        m_Context->setInputShape(name, bindings.dims);
        m_Context->setTensorAddress(name, d_ptr);
#endif
    }

    for (auto& bindings : m_outputs) {
        void* d_ptr, * h_ptr;
        size_t size = bindings.total_bytes;
        CUDA_CHECK(cudaMallocAsync(&d_ptr, size, m_Stream));
        CUDA_CHECK(cudaHostAlloc(&h_ptr, size, 0));
        bindings.d_ptr = d_ptr;
        bindings.h_ptr = d_ptr;

#ifdef TRT_10
        auto name = bindings.name.c_str();
        m_Context->setTensorAddress(name, d_ptr);
#endif
    }
}

void TensorRTInferenceEngine::InferInternal()
{
#ifdef TRT_10
    m_Context->enqueueV3(m_Stream);
#else
    //m_Context->enqueueV2(d_inputPtrs.data(), m_Stream, nullptr); // error usage
#endif
    cudaStreamSynchronize(m_Stream);
}

void TensorRTInferenceEngine::Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity != Severity::kINFO)
        std::cout << "[TensorRT] " << msg << std::endl;
}
