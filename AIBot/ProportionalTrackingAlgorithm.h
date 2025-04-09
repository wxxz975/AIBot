#pragma once

#include "TrackingAlgorithm .h"




class ProportionalTrackingAlgorithm : public TrackingAlgorithm {
public:
    ProportionalTrackingAlgorithm(float kp) : m_kp(kp) {};
    ~ProportionalTrackingAlgorithm() = default;
    
    std::pair<int, int> ComputeNextPosDelta(int currentX, int currentY,
        int targetX, int targetY) override;

private:
    float m_kp;
};

