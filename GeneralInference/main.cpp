#include <iostream>
#include <regex>

#include "InferenceEngine.h"



int main(int argc, char* argv[])
{
    /*
    if (argc != 2) {
        std::cout << "Usage:" << argv[0] << " <model_path>\n";
        return -1;
    }

    std::string model_path = argv[1];
    */
    std::string model_path = "e:\\workspace\\weights\\yolov5\\yolov5n.onnx";
   
    GeneralInference::InferenceEngine engine = GeneralInference::InferenceEngine(model_path);
   
    
   
    cv::VideoCapture cap(0);
    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        std::vector<BoundingBox> boxes = engine.Infer(frame);
        
        cv::Mat rendered = engine.RenderBox(frame, boxes);

        cv::imshow("test", rendered);
        cv::waitKey(10);
    }
    

    return 0;
}