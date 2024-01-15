#include "MoveAdp.h"
#include <Windows.h>
#include "Logger.h"

MovaAdp::MovaAdp()
{
}

MovaAdp::~MovaAdp()
{
}

bool MovaAdp::IsValid() const
{
	return m_resetAllAddr && m_runOnceAddr;
}

bool MovaAdp::Init()
{
	HMODULE hDll = LoadLibraryA(dllName.c_str());
	if (hDll == INVALID_HANDLE_VALUE) {
		err("Failed to load DLL:%s\n", dllName.c_str());
		return false;
	}
	
	m_resetAllAddr = reinterpret_cast<ResetAll>(GetProcAddress(hDll, "ResetAll"));
	m_runOnceAddr = reinterpret_cast<RunOnce>(GetProcAddress(hDll, "RunOnce"));
	
	log("m_resetAllAddr:%llx, m_runOnceAddr:%llx\n",
		m_resetAllAddr,
		m_runOnceAddr);

	return IsValid();
}

BestSolver MovaAdp::CalcCursorDelta(int targetX, int targetY, int curX, int curY) const
{
	return m_runOnceAddr(targetX, targetY, curX, curY);
}

void MovaAdp::ResetAllParams()
{
	m_resetAllAddr();
}
