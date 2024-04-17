#include "OnnxruntimeEngine.h"
#include "Filesystem.h"

#include <cstdint>

namespace InferenceEngine
{
    namespace Engines
    {
        OnnxruntimeEngine::OnnxruntimeEngine()
        {

        }
        OnnxruntimeEngine::~OnnxruntimeEngine()
        {

        }

        bool OnnxruntimeEngine::Initialize(EngineConfig&& config)
        {
            if(!IFilesystem::IsExist(config.model_path)) {
                    
                return false;
            }
                
            
            IFilesystem::GetFilename(config.model_path);
        }

        RawOutputPtrVec OnnxruntimeEngine::Infer(ImageDataPtr data)
        {
            std::vector<Ort::Value> input_tensor = CreateInputTensor(data);
            std::vector<Ort::Value> output_tensor = m_ort_session->Run(Ort::RunOptions{nullptr},
                    m_input_names_ptr.data(),
                    input_tensor.data(),
                    input_tensor.size(),
                    m_output_names_ptr.data(),
                    m_output_names_ptr.size()
                );
            

            return ConvertOutputTensor(output_tensor);
        }

        std::vector<Ort::Value> OnnxruntimeEngine::CreateInputTensor(ImageDataPtr data)
        {
            ONNXTensorElementDataType type = ConvertType(data->type);
            std::vector<int64_t> shape;
            return {
                Ort::Value::CreateTensor(
                    m_ort_memory_info,
                    data->as<void>(),
                    data->data.size(),
                    shape.data(),
                    shape.size(),
                    type
                )
            };
        }

        RawOutputPtrVec OnnxruntimeEngine::ConvertOutputTensor(const std::vector<Ort::Value>& tensors)
        {
            RawOutputPtrVec ret;
            for(const auto& tensor : tensors) 
            {   
                auto tensorInfo = tensor.GetTensorTypeAndShapeInfo();
                std::vector<int64_t> shape = tensorInfo.GetShape();
                ImageEleType type = ConvertType(tensorInfo.GetElementType());
                const void* data = tensor.GetTensorData<void>();

                ret.push_back(std::make_shared<RawOutput>(type, shape, data));
            }

            return ret;
        }

        ONNXTensorElementDataType OnnxruntimeEngine::ConvertType(ImageEleType type)
        {
            switch (type)
            {
            case ImageEleType::u8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            case ImageEleType::f32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            case ImageEleType::f16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            case ImageEleType::i64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            default:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            }
        }

        ImageEleType OnnxruntimeEngine::ConvertType(ONNXTensorElementDataType type)
        {
            switch (type)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return ImageEleType::u8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return ImageEleType::f32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return ImageEleType::f16;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return ImageEleType::i64;
            default:
                return ImageEleType::f32;
            }
        }

    } // namespace Engines
    
} // namespace InferenceEngine
