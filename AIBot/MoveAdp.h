#pragma once
#include <string>

struct BestSolver
{
	int dx, dy;
};


typedef void (*ResetAll)();

typedef BestSolver(*RunOnce)(int targetX, int targetY, int curX, int curY);



class MovaAdp
{
public:
	MovaAdp();
	~MovaAdp();

	bool IsValid() const;

	bool Init();

	BestSolver CalcCursorDelta(int targetX, int targetY, int curX, int curY) const;
	
	void ResetAllParams();

private:
	std::string dllName = "MoveAdaptor.dll";
	ResetAll m_resetAllAddr = nullptr;
	RunOnce m_runOnceAddr = nullptr;
};

