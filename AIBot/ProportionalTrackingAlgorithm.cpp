#include "ProportionalTrackingAlgorithm.h"





std::pair<int, int> ProportionalTrackingAlgorithm::ComputeNextPosDelta(int currentX, int currentY,
    int targetX, int targetY) {
    int deltaX = targetX - currentX;
    int deltaY = targetY - currentY;
    return std::make_pair(deltaX * m_kp, deltaY * m_kp);
}   

