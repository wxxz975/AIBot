#pragma once

#include <cstdint>
#include <string>

enum class ModelType: int
{
	YOLOv8
};
enum class ModelFormat : int
{
	ONNX
};

enum class MouseInputType: int
{
	WINDOWS_API_INPUT,
	LOGI_API_INPUT,
	ARDUINO_CUSTOM_INPUT
};

enum class MouseMoveAlgorithm
{
	PROPORTIONAL_TRACKING,
	PID_TRACKING
};

struct BotConfig
{

	// 模型相关参数
	std::string model_path;
	const ModelType type = ModelType::YOLOv8;
	const ModelFormat format = ModelFormat::ONNX;
	float iou_threshold;
	float conf_threshold;


	MouseInputType mouse_input_type;
	MouseMoveAlgorithm mouse_move_algo;

	union {
		struct 
		{
			// pid系数
			float x_kp;
			float x_ki;
			float x_kd;

			float y_kp;
			float y_ki;
			float y_kd;
		}pid_args;

		struct 
		{
			float alpha;
		}proportion_arg;
	}mouse_move_args;


};