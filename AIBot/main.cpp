

/*
	这里负责程序的主入口、主要是做将各个模块组合起来
*/

#include "Inference.h"
#include "MouseEv.h"
#include "ScreenShot.h"
#include "Logger.h"
#include "MoveAdp.h"

#include <crtdbg.h>
#include <Windows.h>
#include "Filesystem.h"


// 截取左面中心图像的宽高
const int width = 640;
const int height = 640;

const int DeskWidth = 2560;
const int DeskHeight = 1440;

const int OriginX = (DeskWidth - width) / 2;
const int OriginY = (DeskHeight - height) / 2;

bool isAimActive = false;

const float MAX_AIM_DISTENCE = 400; // 最大瞄准距离


#define IS_KEY_PRESSED(key) (GetAsyncKeyState(key) & 0x8001)

void ShowInference(const std::vector<ResultNode>& result)
{
	for (const auto& node : result)
		log("x:%f, y:%f, w:%f, h:%f, classIdx:%d, confidence:%f\n",
			node.x, node.y, node.w, node.h, node.classIdx, node.confidence);
}


/// <summary>
/// 找出距离坐标x,y最近的object的中心点
/// </summary>
/// <param name="targets"></param>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
std::pair<int, float> findNearestTarget(const std::vector<ResultNode>& targets, float x, float y) {
	int nearestIndex = -1;
	float nearestDistance = std::numeric_limits<float>::infinity();

	for (std::size_t idx = 0; idx < targets.size(); ++idx) {
		float center_x = targets[idx].x + OriginX + targets[idx].w / 2;
		float center_y = targets[idx].y + OriginY + targets[idx].h / 2;

		float distance = std::sqrt(std::pow(x - center_x, 2) + std::pow(y - center_y, 2));

		if (distance < nearestDistance) {
			nearestIndex = idx;
			nearestDistance = distance;
		}
	}

	return std::make_pair(nearestIndex, nearestDistance);
}

bool InitDllAPI()
{

}

int main(int argc, char* argv[])
{


	MouseEv mouse;
	ScreenShot shotter;
	Inference engine;
	MovaAdp adp;
	POINT cursor = {DeskWidth/2, DeskHeight/2 };

	std::string executable_dir = IFilesystem::GetExecutablePath(argv[0]);
	log("executable_directory:%s\n", executable_dir.c_str());
	
	const std::string model_path = IFilesystem::ConcatPath(executable_dir, "models\\csgo_yolov5n_10w.onnx");
	
	log("model_path:%s\n", model_path.c_str());

	if (!mouse.Init() || !adp.Init() || !shotter.Init(width, height) || !engine.Init(model_path))
	{
		err("failed to init!\n");
		return -1;
	}
	
	
	log("start gaming!\n");
	
	//cv::VideoCapture cap("E:\\workspace\\games\\CSGO\\datasets\\783311450-1-208.mp4");
	
	while (true)
	{
		if (IS_KEY_PRESSED(VK_END))
			break;
		

		cv::Mat img = shotter.Capture();
		if (img.empty())
			continue;

		auto start_time = std::chrono::high_resolution_clock::now();
		const std::vector<ResultNode>& result = engine.doInfer(img);
		auto end_time = std::chrono::high_resolution_clock::now();

		
		if (result.empty()) 
			continue;

		//ShowInference(result);

		/*
		if (!GetCursorPos(&cursor))
			continue;
		*/
		

		
		std::pair<int, float> bestTarInfo = findNearestTarget(result, cursor.x, cursor.y);
		const auto& bestTarget = result[bestTarInfo.first];

		log("Nearest distance:%f\n", bestTarInfo.second);
		// 距离必须小于阈值
		if (bestTarInfo.second > MAX_AIM_DISTENCE)
			continue;

		/*
		if (!IS_KEY_PRESSED(VK_RBUTTON)) {
			isAimActive = false;
			continue;
		}*/


		if(isAimActive)
			adp.ResetAllParams();

		// 计算鼠标指针到目标之间需要移动的delta
		int targetX = OriginX + bestTarget.x + bestTarget.w / 2;
		int targetY = OriginY + bestTarget.y + bestTarget.h / 4; // 除以2 正中央， 除以3 偏上， 除得越大越靠近头顶
		auto resolve = adp.CalcCursorDelta(targetX, targetY, cursor.x, cursor.y);
		
		
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		log("inferTime:%ld ms, best resolve dx:%d, dy:%d, targetX:%d, targetY:%d, curX:%d, curY:%d\n",
			duration.count(),
			resolve.dx, resolve.dy, targetX, targetY, cursor.x, cursor.y);


		mouse.Move(resolve.dx, resolve.dy);
	}

	return 0;
}