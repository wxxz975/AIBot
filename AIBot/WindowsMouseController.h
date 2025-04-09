#pragma once

#include "IMouseController .h"




class WindowsMouseController : public IMouseController {
public:
    WindowsMouseController() = default;
    ~WindowsMouseController() = default;

    MouseMoveStatus MoveMouse(int delta_x, int delta_y) override;
};
