#pragma once

#include <vector>


// TrackingAlgorithm is an abstract class that defines the interface for a tracking algorithm, 
// the next position delta is computed based on the current position and the target position
class TrackingAlgorithm {
public:
    TrackingAlgorithm() = default;
    virtual ~TrackingAlgorithm() = default;


    /*
        Compute the next position delta
        @param currentX: the current x position
        @param currentY: the current y position
        @param targetX: the target x position
        @param targetY: the target y position
        @return: the next position delta
    */ 
    virtual std::pair<int, int> ComputeNextPosDelta(int currentX, int currentY,
        int targetX, int targetY) = 0;
};


enum TrackingAlgorithmType
{
    Proportional,
    PID
};