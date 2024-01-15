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
/// �������PID�㷨���з�װ�� ����x���y�����������ϵļ���
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
	PID_Calibration cali_x = { 0.2, 0.04, 0.1}; // ƽ��
	PID_Calibration cali_y = { 0.2, 0.04, 0.1};

	PID_State state_x = {};
	PID_State state_y = {};

	//POINT cursorPos;
};
