#pragma once
#include "MoveAdaptor.h"



#define DLL_EXPORT __declspec(dllexport)


static MoveAdaptor adaptor;

extern "C" {
	
	inline DLL_EXPORT void ResetAll();
	
	inline DLL_EXPORT BestSolver RunOnce(int targetX, int targetY, int curX, int curY);
}

inline DLL_EXPORT void ResetAll()
{
	adaptor.ResetAll();
}

inline DLL_EXPORT BestSolver RunOnce(int targetX, int targetY, int curX, int curY)
{
	return adaptor.RunOnce(targetX, targetY, curX, curY);
}
