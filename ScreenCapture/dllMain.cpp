#include "Export.h"

/*
	负责图像截取、改为动态链接库形式
	目前使用dxgi + directX的方式进行抓取， 这个方式可以在抓取的时候设置等待的时间。
*/




BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        
        break;
    }

    return TRUE;
}