#pragma once

#include <string>



class MouseEv
{
public:
	typedef bool(*Initialize)(); // 初始化dll链接库
	typedef void(*Deinitialize)(); // 

	// char button, char x, char y, char wheel // 调用鼠标移动处
	typedef void(*Event)(char button, char x, char y, char wheel);
public:
	MouseEv();
	~MouseEv();

	bool IsValid() const;

	bool Move(char dx, char dy) const;

	bool Init();

private:
	std::string dllName = "MouseEvent.dll";

	Initialize m_initAddr		= nullptr;
	Deinitialize m_deinitAddr	= nullptr;
	Event m_eventAddr			= nullptr;
};
