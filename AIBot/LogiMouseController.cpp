#include "LogiMouseController.h"

#include <winternl.h>
#pragma comment(lib, "ntdll.lib")

LogiMouseController::LogiMouseController()
{
}

LogiMouseController::~LogiMouseController()
{
}

bool LogiMouseController::Initialze()
{
	m_FoundMouse = NT_SUCCESS(InitDevice(m_DeviceName1)) || NT_SUCCESS(InitDevice(m_DeviceName2));
	return m_FoundMouse;
}

void LogiMouseController::Deinitialize()
{
	if (m_HInput != 0) {
		NtClose(m_HInput);
		m_HInput = 0;
	}
}

bool LogiMouseController::Event(char button, char x, char y, char wheel)
{
	MOUSE_IO io = { 0 };
	io.unk1 = 0;
	io.button = button;
	io.x = x;
	io.y = y;
	io.wheel = wheel;


	if (!DoEvent(&io)) {
		Deinitialize();
		return false;
	}

	return true;
}

MouseMoveStatus LogiMouseController::MoveMouse(int delta_x, int delta_y)
{
	return Event(0, delta_x, delta_y, 0) ? MouseMoveStatus::OK : MouseMoveStatus::DRIVER_FAILURE;
}

bool LogiMouseController::InitDevice(const std::wstring& device_name)
{
	UNICODE_STRING name;
	OBJECT_ATTRIBUTES attr = {};
	IO_STATUS_BLOCK io_status;

	RtlInitUnicodeString(&name, device_name.c_str());
	InitializeObjectAttributes(&attr, &name, 0, NULL, NULL);

	NTSTATUS status = NtCreateFile(&m_HInput, GENERIC_WRITE | SYNCHRONIZE, &attr, &io_status, 0,
		FILE_ATTRIBUTE_NORMAL, 0, 3, FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, 0, 0);

	return NT_SUCCESS(status);
}

bool LogiMouseController::DoEvent(MOUSE_IO* buf)
{
	IO_STATUS_BLOCK block;
	return NT_SUCCESS(NtDeviceIoControlFile(m_HInput, 0, 0, 0, &block, 0x2a2010, buf, sizeof(MOUSE_IO), 0, 0));
}
