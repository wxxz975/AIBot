#include "MouseMove.h"


#include <winternl.h>
#pragma comment(lib, "ntdll.lib")

bool MouseMove::Initialze()
{
	found_mouse = NT_SUCCESS(InitDevice(device_name_1)) || NT_SUCCESS(InitDevice(device_name_2));
	return found_mouse;
}

void MouseMove::Deinitialize()
{
	if (hInput != 0) {
		NtClose(hInput);
		hInput = 0;
	}
}

void MouseMove::Event(char button, char x, char y, char wheel)
{
	MOUSE_IO io = { 0 };
	io.unk1 = 0;
	io.button = button;
	io.x = x;
	io.y = y;
	io.wheel = wheel;

	if (!DoEvent(&io)) {
		Deinitialize();
		Initialze();
	}
}

bool MouseMove::InitDevice(const std::wstring& device_name)
{
	UNICODE_STRING name;
	OBJECT_ATTRIBUTES attr = {};
	IO_STATUS_BLOCK io_status;

	RtlInitUnicodeString(&name, device_name.c_str());
	InitializeObjectAttributes(&attr, &name, 0, NULL, NULL);

	NTSTATUS status = NtCreateFile(&hInput, GENERIC_WRITE | SYNCHRONIZE, &attr, &io_status, 0,
		FILE_ATTRIBUTE_NORMAL, 0, 3, FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, 0, 0);

	return NT_SUCCESS(status);
}

bool MouseMove::DoEvent(MOUSE_IO* buf)
{
	IO_STATUS_BLOCK block;
	return NtDeviceIoControlFile(hInput, 0, 0, 0, &block, 0x2a2010, buf, sizeof(MOUSE_IO), 0, 0) == 0L;
}
