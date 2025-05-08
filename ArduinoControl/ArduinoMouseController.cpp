#include "ArduinoMouseController.h"

#include <iostream>


ArduinoMouseController::ArduinoMouseController()
    :m_hSerial(nullptr)
{

}

bool ArduinoMouseController::Initialize(const std::string& portname)
{
    m_portName = std::wstring(portname.begin(), portname.end());
    m_hSerial = CreateFile(m_portName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        0,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        0);

    if (m_hSerial == INVALID_HANDLE_VALUE) {
        std::cerr << "Error opening serial port" << std::endl;
        return false;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

    if (!GetCommState(m_hSerial, &dcbSerialParams)) {
        std::cerr << "Error getting state" << std::endl;
        CloseHandle(m_hSerial);
        return false;
    }

    dcbSerialParams.BaudRate = CBR_115200;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;

    if (!SetCommState(m_hSerial, &dcbSerialParams)) {
        std::cerr << "Error setting state" << std::endl;
        CloseHandle(m_hSerial);
        return false;
    }

    return true;
}

MouseMoveStatus ArduinoMouseController::MoveMouse(int delta_x, int delta_y)
{
    COMPackage pkg{};
    pkg.type = CommandType::CMD_MOUSE_MOVE;
    pkg.InitMoveArg(delta_x, delta_y);
   
    return WriteData(&pkg, sizeof(COMPackage)) ? MouseMoveStatus::OK : MouseMoveStatus::INVALID_INPUT;
}

void ArduinoMouseController::CloseCOM()
{
    if (m_hSerial != INVALID_HANDLE_VALUE) {
        CloseHandle(m_hSerial);
        m_hSerial = INVALID_HANDLE_VALUE;
    }
}

bool ArduinoMouseController::WriteData(void* data, size_t size) const
{
    DWORD bytesWritten;
    if (!WriteFile(m_hSerial, data, size, &bytesWritten, NULL)) {
        
        std::cerr << "Error writing to serial port" << std::endl;
    }

    char buffer[64] = {0};
    bool ret = ReadFile(m_hSerial, buffer, sizeof(buffer), nullptr, nullptr);
    std::cout << buffer << "\n";

    return bytesWritten == size;
}
