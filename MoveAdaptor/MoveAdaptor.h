#pragma once
#include <vector>
#include <Windows.h>
#include "../Inference/YoloDefine.h"

#include "PID.h"

struct BestSolver
{
	int dx, dy;
};


/// <summary>
/// 这里采用PID算法进行封装， 包含x轴和y轴两个方向上的计算
/// </summary>
class MoveAdaptor
{
public:
	MoveAdaptor();
	~MoveAdaptor();


	void ResetAll();

	

	//std::pair<int, int> GetBestTarget();

	BestSolver RunOnce(int targetX, int targetY, int curX, int curY);
private:

	void ResetState(PID_State& state);

private:
	PID_Calibration cali_x = { 0.2, 0.04, 0.1}; // 平衡
	PID_Calibration cali_y = { 0.2, 0.04, 0.1};

	PID_State state_x = {};
	PID_State state_y = {};

	//POINT cursorPos;
};
