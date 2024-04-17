#pragma once
#include <memory>

#include "InferenceEngine.h"

#include <onnxruntime_cxx_api.h>


namespace InferenceEngine
{
    namespace Engines
    {
        
        class OnnxruntimeEngine: public InferenceEngine
        {
        public:
            OnnxruntimeEngine();
            ~OnnxruntimeEngine();

            bool Initialize(EngineConfig&& config) override;

            RawOutputPtrVec Infer(ImageDataPtr data) override;

        private:
            std::vector<Ort::Value> CreateInputTensor(ImageDataPtr data);

            RawOutputPtrVec ConvertOutputTensor(const std::vector<Ort::Value>& tensor);

            ONNXTensorElementDataType ConvertType(ImageEleType type);

            ImageEleType ConvertType(ONNXTensorElementDataType type);
            
            //std::vector ConvertShape();


        private:
            Ort::Env m_ort_env;

            std::shared_ptr<Ort::Session> m_ort_session;
            Ort::MemoryInfo m_ort_memory_info{nullptr};


            std::vector<std::string> m_input_names;
            std::vector<const char*> m_input_names_ptr;


            std::vector<std::string> m_output_names;
            std::vector<const char*> m_output_names_ptr;
        };
        
        
        


    }; // namespace Engines
};