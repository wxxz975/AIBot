#pragma once
#include <string>


class TensorRTInferenceEngine;
class TextureMapper;
class DesktopTextureCapturer;
struct BotConfig;

class InferenceContext
{
public:
	InferenceContext() = default;
	~InferenceContext() = default;

	bool Initialize(BotConfig config);
	// 1、获取图像
	//GetNext();


	//2、调用tensorrt推理


	//3、清洗获取结果

private:

	TensorRTInferenceEngine* m_engine;
	
	DesktopTextureCapturer* m_capture;
	TextureMapper* m_mapper;
};
