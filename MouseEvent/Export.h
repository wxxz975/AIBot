#pragma once
#include "MouseMove.h"


#define DLL_EXPORT __declspec(dllexport)

extern "C" {
    DLL_EXPORT bool Initialize();
    DLL_EXPORT void Deinitialize();
    DLL_EXPORT void Event(char button, char x, char y, char wheel);

}


static MouseMove mover;

static bool initailized = false;

// 初始化函数
inline DLL_EXPORT bool Initialize()
{
    return mover.Initialze();
}

// 反初始化函数
inline DLL_EXPORT void Deinitialize()
{
    mover.Deinitialize();
}

// 事件处理函数
inline DLL_EXPORT void Event(char button, char x, char y, char wheel)
{
    mover.Event(button, x, y, wheel);
}

