#pragma once

#include <string>
#include <Windows.h>

// 这个是使用老版本的logi驱动控制鼠标移动

typedef struct {
	char button;
	char x;
	char y;
	char wheel;
	char unk1;
} MOUSE_IO;

#define MOUSE_PRESS 1
#define MOUSE_RELEASE 2
#define MOUSE_MOVE 3
#define MOUSE_CLICK 4

class MouseMove
{
public:
	bool Initialze();

	void Deinitialize();

	void Event(char button, char x, char y, char wheel);

	/// <summary>
	/// 
	/// </summary>
	//void Move()

private:
	bool InitDevice(const std::wstring& device_name);

	bool DoEvent(MOUSE_IO* buf);

private:

	std::wstring device_name_1 = L"\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
	std::wstring device_name_2 = L"\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}";

	HANDLE hInput = 0;

	bool found_mouse = false;
};



