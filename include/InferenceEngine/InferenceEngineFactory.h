#pragma once

#include <memory>



namespace InferenceEngine
{
    class OpenVINOEngine;
    class OnnxruntimeEngine;

    
    class InferenceEngineFactory
    {
    public:;
        virtual ~InferenceEngineFactory() = default;

        virtual std::unique_ptr<OnnxruntimeEngine> CreateOnnxruntimeEngine() = 0;

        virtual std::unique_ptr<OpenVINOEngine> CreateOpenVINOEngine() = 0;



    };
    
    

};