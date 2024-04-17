#pragma once
#include "InferenceEngine.h"

#include <openvino/openvino.hpp>


namespace InferenceEngine
{
    namespace Engines
    {
        
        class OpenVINOEngine: public InferenceEngine
        {
        public:
            OpenVINOEngine();
            ~OpenVINOEngine();

            bool Initialize(EngineConfig&& config) override;

            RawOutputPtrVec Infer(ImageDataPtr data) override;

        private:

            std::vector<ov::Tensor> CreateInputTensor(ImageDataPtr data);

            RawOutputPtrVec ConvertOutputTensor(const ov::Tensor& tensor);

            ov::element::Type_t ConvertEleType(ImageEleType type);

            ImageEleType ConvertEleType(ov::element::Type_t type);

            inline std::vector<std::size_t> ConvertShape(const ov::Shape& shape)
            {
                std::vector<std::size_t> ret;
                for(const auto& s : shape)
                    ret.push_back(s);
                
                return ret;
            };

        private:
            std::shared_ptr<ov::Model> m_ov_model;
            ov::CompiledModel m_ov_compiled_model;
            
            ov::InferRequest m_ov_infer_request;
        };
        
        
        


    } // namespace Engines
};