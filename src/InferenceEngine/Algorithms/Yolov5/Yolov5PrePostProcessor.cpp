#include "Yolov5PrePostProcessor.h"


namespace InferenceEngine
{
    namespace Algorithms
    {
        Yolov5PrePostProcessor::Yolov5PrePostProcessor(/* args */)
        {
        }
        
        Yolov5PrePostProcessor::~Yolov5PrePostProcessor()
        {
        }

        ImageDataPtr Yolov5PrePostProcessor::Preprocessing(const cv::Mat& image)
        {

        }

        std::vector<BoundingBox> Yolov5PrePostProcessor::Postprocessing(RawOutputPtrVec output)
        {
            
        }
    }
}