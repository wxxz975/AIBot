#include <Windows.h>
#include <chrono>
#include <thread>
#include <iostream>

/*
	��������������ƶ����⡣
							����ƶ��� (ֱ���ƶ���Ŀ�����ϡ������ƶ������ձ����ƶ���PID����)
*/


#include "Export.h"


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
/*
int main(int argc, char* argv[])
{
	int target_x = 500;
	int target_y = 500;

	try
	{
		std::thread th([&]() {
			while (true)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));

				if (target_x > 2000)
					target_x = 500;
				if (target_y < 100)
					target_y = 500;

				target_x += 10;
				target_y -= 10;
			}
			});

		MoveAdaptor mover;


		while (true) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			mover.RunOnce(target_x, target_y);
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "err:" << e.what() << "\n";
	}
	

	

	return 0;
}

*/