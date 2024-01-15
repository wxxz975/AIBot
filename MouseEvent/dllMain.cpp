#include "Export.h"

/*
	负责鼠标移动模块(使用logi是为了隐藏调用鼠标的接口)
    注: 
        1、当安装完成之后先等程序安装你所使用的鼠标的扩展，装好之后立马断开网络连接，然后在设置界面取消自动升级
        2、如果在测试界面发现鼠标移动无效或失效，可以尝试先退出GHub，后使用管理员权限打开GHub重新测试
*/



BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        Initialize();
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        Deinitialize();
        break;
    }

    return TRUE;
}