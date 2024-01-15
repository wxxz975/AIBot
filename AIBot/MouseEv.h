#pragma once

#include <string>



class MouseEv
{
public:
	typedef bool(*Initialize)(); // ��ʼ��dll���ӿ�
	typedef void(*Deinitialize)(); // 

	// char button, char x, char y, char wheel // ��������ƶ���
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
