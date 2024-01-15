#include "ScreenShot.h"

#include <Windows.h>

#include "Logger.h"

ScreenShot::ScreenShot()
{
}

ScreenShot::~ScreenShot()
{
	//if (IsValid)
		//FreeLibrary(hDll);
}

bool ScreenShot::Init(int width, int height)
{
	HMODULE hDll = LoadLibraryA(dllName.c_str());
	if (hDll == INVALID_HANDLE_VALUE)
	{
		err("Failed to load DLL:%llx\n", dllName.c_str());
		return false;
	}

	m_initAddr =  reinterpret_cast<Initialize>(GetProcAddress(hDll, "Initialize"));
	m_capAddr = reinterpret_cast<CaptureNext>(GetProcAddress(hDll, "CaptureNext"));

	log("m_initAddr:%llx, m_capAddr:%llx\n", m_initAddr, m_capAddr);
	img = cv::Mat(width, height, CV_8UC4);

	return IsValid() && m_initAddr(width, height);
}

bool ScreenShot::IsValid() const
{
	return m_initAddr && m_capAddr;
}

const cv::Mat& ScreenShot::Capture()
{
	
	if(m_capAddr(img))
		return img;

	//log("Failed to capture!\n"); 这个可能不存在

	return cv::Mat();
}
