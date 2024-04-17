#pragma once

#include "General.h"


namespace InferenceEngine
{
    struct EngineConfig
    {
        std::string model_path;
        float conf_threshold;
        float iou_threshold;
        // ....
    };
    

    class InferenceEngine
    {
    public:
        InferenceEngine();
        ~InferenceEngine();

        virtual bool Initialize(EngineConfig&& config) = 0;

        virtual RawOutputPtrVec Infer(ImageDataPtr data) = 0;
    };
    
    
    

};