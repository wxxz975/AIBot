#pragma once

#include "IMouseController .h"

#include <string>
#include <Windows.h>


/*
	TODO: 这个需要arduino板子，其次这个可以选择性的进行简易的加密
*/ 
class ArduinoMouseController: public IMouseController
{
public:
	ArduinoMouseController();
	~ArduinoMouseController() = default;

	bool Initialize(const std::string& portname);

	MouseMoveStatus MoveMouse(
		int delta_x, int delta_y
	) override;


	enum class CommandType: int32_t
	{
		CMD_INIT_DEVICE = 0x1,
		CMD_MOUSE_MOVE = 0x2,
		CMD_CLOSE_DEVICE = 0x3,
	};
	struct COMPackage
	{
		enum CommandType type;
		union PackageContent
		{
			struct 
			{
				int32_t padding;
			}init_arg;

			struct 
			{
				int32_t delta_x;
				int32_t delta_y;
			}move_arg;

			struct 
			{
				int32_t padding;
			}close_arg;
		}Content;

		void InitMoveArg(int32_t delta_x, int32_t delta_y) {
			Content.move_arg.delta_x = delta_x;
			Content.move_arg.delta_y = delta_y;
		}

		void InitDeviceArg(int32_t arg) {
			Content.init_arg.padding = arg;
		}
	};
	

private:
	void CloseCOM();

	bool WriteData(void* data, size_t size) const;

private:
	std::wstring m_portName;
	HANDLE m_hSerial;
};