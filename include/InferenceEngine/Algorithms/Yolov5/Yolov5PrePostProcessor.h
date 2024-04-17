#pragma once

#include "PrePostProcessor.h"

namespace InferenceEngine
{
    namespace Algorithms
    {
        class Yolov5PrePostProcessor: public PrePostProcessor
        {
        public:
            Yolov5PrePostProcessor(/* args */);
            ~Yolov5PrePostProcessor();

            ImageDataPtr Preprocessing(const cv::Mat& image) override;

            std::vector<BoundingBox> Postprocessing(RawOutputPtrVec output) override;
        };
        
        
        
    } // namespace Algorithms
    
} // namespace InferenceEngine
