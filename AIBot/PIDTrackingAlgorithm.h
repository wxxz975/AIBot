#pragma once

#include "TrackingAlgorithm .h"
#include "PID.h"

class PIDTrackingAlgorithm : public TrackingAlgorithm {
public:
    PIDTrackingAlgorithm(PID_Calibration x_cali, PID_Calibration y_cali);
    ~PIDTrackingAlgorithm() = default;

    std::pair<int, int> ComputeNextPosDelta(int currentX, int currentY,
        int targetX, int targetY) override;

    void ResetState();  // 重置PID状态

private:
    PID_State m_xState;  // X方向的PID状态
    PID_State m_yState;  // Y方向的PID状态

    PID_Calibration m_x_cali;
    PID_Calibration m_y_cali;
};

