#include "MoveAdaptor.h"



MoveAdaptor::MoveAdaptor()
{
	ResetAll();
}

MoveAdaptor::~MoveAdaptor()
{
}

void MoveAdaptor::ResetAll()
{
	ResetState(state_x);
	ResetState(state_y);
}

void MoveAdaptor::ResetState(PID_State& state)
{
	state.actual = 0.0;
	state.target = 0.0;
	state.time_delta = 1.0; // assume an arbitrary time interval of 1.0
	state.previous_error = 0.0;
	state.integral = 0.0;
}


BestSolver MoveAdaptor::RunOnce(int targetX, int targetY, int curX, int curY)
{
	state_x.target = targetX;
	state_y.target = targetY;

	state_x.actual = curX;
	state_y.actual = curY;

	state_x = pid_iterate(cali_x, state_x);
	state_y = pid_iterate(cali_y, state_y);

	//return { static_cast<long>((targetX - curX) / 1.f), static_cast<long>((targetY - curY) / 1.f) };
	return { static_cast<long> (state_x.output), static_cast<long>(state_y.output) };
}



