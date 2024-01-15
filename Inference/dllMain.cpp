#include "Export.h"

#include <opencv2/opencv.hpp>

#include <Windows.h>
#include <iostream>

//#define log printf


/*
	此模块使用onnx部署yolov5、并且制作为动态链接库或静态链接库的形式
*/
/*
void ShowInference(const std::vector<ResultNode>& result)
{
    for (const auto& node : result)
        log("x:%f, y:%f, w:%f, h:%f, classIdx:%d, confidence:%f\n",
            node.x, node.y, node.w, node.h, node.classIdx, node.confidence);
}*/
/*
#define IS_KEY_PRESSED(key) (GetAsyncKeyState(key) & 0x8001)
int main(int argc, char* argv[])
{

    std::string model_path = "E:\\workspace\\games\\AIBot\\x64\\Release\\models\\csgo_yolov5n_10w.onnx";
    std::string video_path = "E:\\workspace\\games\\CSGO\\datasets\\783311450-1-208.mp4";

    InferenceEngine engine;
    engine.Initialize(model_path);
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
        return 0;
    
    cv::Mat img;

    while (cap.read(img))
    {
        if (IS_KEY_PRESSED(VK_END))
            break;

        auto result = engine.Infer(img);
        
        //ShowInference(result);

        cv::Mat renderd = engine.Render(img, result);
        
        cv::imshow("Inference Test!", renderd);
        cv::waitKey(10);
    }

    std::cout << "endl\n";
}
*/

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        break;
    }

    return TRUE;
}