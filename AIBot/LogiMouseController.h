#pragma once

#include <string>
#include <Windows.h>

#include "IMouseController .h"


class LogiMouseController: public IMouseController
{

public:
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

public:
	LogiMouseController();
	~LogiMouseController();

	bool Initialze();

	void Deinitialize();

	bool Event(char button, char x, char y, char wheel);

	MouseMoveStatus MoveMouse(int delta_x, int delta_y) override;

private:
	bool InitDevice(const std::wstring& device_name);

	bool DoEvent(MOUSE_IO* buf);

private:
	std::wstring m_DeviceName1 = L"\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
	std::wstring m_DeviceName2 = L"\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}";

	HANDLE m_HInput = 0;

	bool m_FoundMouse = false;

};

