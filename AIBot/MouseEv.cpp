#include "MouseEv.h"


#include <Windows.h>

#include "Logger.h"

MouseEv::MouseEv()
{
}

MouseEv::~MouseEv()
{
	if(IsValid())
		m_deinitAddr();
}

bool MouseEv::IsValid() const
{
	return m_initAddr && m_deinitAddr && m_eventAddr;
}

bool MouseEv::Move(char dx, char dy) const
{
	
	if (!IsValid()) {
		log("Not initialize the MouseEv!\n");
		return false;
	}

	m_eventAddr(0, dx, dy, 0);

	//mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0);

	return true;
}

bool MouseEv::Init()
{
	HMODULE hDll = LoadLibraryA(dllName.c_str());
	if (hDll == INVALID_HANDLE_VALUE)
	{
		err("Failed to load DLL:%s\n", dllName.c_str());
		return false;
	}

	m_initAddr = reinterpret_cast<Initialize>(GetProcAddress(hDll, "Initialize"));
	m_deinitAddr = reinterpret_cast<Deinitialize>(GetProcAddress(hDll, "Deinitialize"));
	m_eventAddr = reinterpret_cast<Event>(GetProcAddress(hDll, "Event"));

	log("m_initAddr:%llx, m_deinitAddr:%llx, m_eventAddr:%llx\n", 
		m_initAddr, 
		m_deinitAddr, 
		m_eventAddr);
	
	return IsValid() && m_initAddr()/*调用dll中的初始化函数*/;
}
