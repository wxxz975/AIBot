#include "Export.h"

/*
	��������ƶ�ģ��(ʹ��logi��Ϊ�����ص������Ľӿ�)
    ע: 
        1������װ���֮���ȵȳ���װ����ʹ�õ�������չ��װ��֮������Ͽ��������ӣ�Ȼ�������ý���ȡ���Զ�����
        2������ڲ��Խ��淢������ƶ���Ч��ʧЧ�����Գ������˳�GHub����ʹ�ù���ԱȨ�޴�GHub���²���
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