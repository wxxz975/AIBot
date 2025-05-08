#pragma once


#include "skCrypter.h"



#define ObfStr(str) skCrypt(str)




#define XOR_KEY __TIME__[4]


inline void XorMem(char* ptr, int size, char key = XOR_KEY)
{
	for (int idx = 0; idx < size; ++idx) 
	{
		ptr[idx] ^= key;
	}
}