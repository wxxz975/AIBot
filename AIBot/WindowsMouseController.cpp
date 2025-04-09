#include "WindowsMouseController.h"

#include <Windows.h>

MouseMoveStatus WindowsMouseController::MoveMouse(int delta_x, int delta_y)
{
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.dx = delta_x;
    input.mi.dy = delta_y;

    if (SendInput(1, &input, sizeof(INPUT)) != 1) {
        return MouseMoveStatus::DRIVER_FAILURE;
    }
    return MouseMoveStatus::OK;
}

