#include "OpenVINOEngine.h"


namespace InferenceEngine
{
    namespace Engines
    {
        OpenVINOEngine::OpenVINOEngine()
        {
            
        }
        
        OpenVINOEngine::~OpenVINOEngine()
        {

        }

        bool OpenVINOEngine::Initialize(EngineConfig&& config)
        {
            ov::Core ov_core;
            m_ov_model = ov_core.read_model(config.model_path);
        
            m_ov_compiled_model = ov_core.compile_model(m_ov_model, "AUTO");

            m_ov_infer_request = m_ov_compiled_model.create_infer_request();

            return true;
        }

        RawOutputPtrVec OpenVINOEngine::Infer(ImageDataPtr data)
        {
            std::vector<ov::Tensor> in_tensor = CreateInputTensor(data);
            m_ov_infer_request.set_input_tensors(in_tensor);
            m_ov_infer_request.infer();

            ov::Tensor out_tensor = m_ov_infer_request.get_output_tensor();
            
            return ConvertOutputTensor(out_tensor);
        }

        std::vector<ov::Tensor> OpenVINOEngine::CreateInputTensor(ImageDataPtr data)
        {
            return {
                ov::Tensor(
                    ConvertEleType(data->type),
                    {1, data->channels, data->height, data->width},
                    data->data
                )
            };
        }

        RawOutputPtrVec OpenVINOEngine::ConvertOutputTensor(const ov::Tensor& tensor)
        {
            ImageEleType type = ConvertEleType(tensor.get_element_type());
            std::vector<std::size_t> shape = ConvertShape(tensor.get_shape());
            void* data = tensor.data();

            return { std::make_shared<RawOutput>(type, shape, data) }; // 这个好像只能处理一个维度的输出
        }

        ov::element::Type_t OpenVINOEngine::ConvertEleType(ImageEleType type)
        {
            switch (type)
            {
            case ImageEleType::u8:
                return ov::element::u8;
            case ImageEleType::f32:
                return ov::element::f32;
            case ImageEleType::f16:
                return ov::element::f16;
            case ImageEleType::i64:
                return ov::element::i64;
            default:
                return ov::element::f32; // default as f32
            }
        }

        ImageEleType OpenVINOEngine::ConvertEleType(ov::element::Type_t type)
        {
            switch (type)
            {
            case ov::element::u8:
                return ImageEleType::u8;
            case ov::element::f32:
                return ImageEleType::f32;
            case ov::element::f16:
                return ImageEleType::f16;
            case ov::element::i64:
                return ImageEleType::i64;
            default:
                ImageEleType::f32;
            }
        }
    } // namespace Engines
};