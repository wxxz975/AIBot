#include "PIDTrackingAlgorithm.h"


// 初始化PID状态
PIDTrackingAlgorithm::PIDTrackingAlgorithm(PID_Calibration x_cali, PID_Calibration y_cali)
    : m_x_cali(x_cali), m_y_cali(y_cali)
{
    ResetState();
}

void PIDTrackingAlgorithm::ResetState() {
    memset(&m_xState, 0, sizeof(PID_State));
    memset(&m_yState, 0, sizeof(PID_State));
}

std::pair<int, int> PIDTrackingAlgorithm::ComputeNextPosDelta(int currentX, int currentY,
    int targetX, int targetY) {
    
    // 更新X方向的状态
    m_xState.actual = static_cast<double>(currentX);
    m_xState.target = static_cast<double>(targetX);
    
    // 更新Y方向的状态
    m_yState.actual = static_cast<double>(currentY);
    m_yState.target = static_cast<double>(targetY);
    
    // 执行PID迭代
    m_xState = pid_iterate(m_x_cali, m_xState);
    m_yState = pid_iterate(m_y_cali, m_yState);
    
    // 将输出转换为整数增量
    int deltaX = static_cast<int>(m_xState.output);
    int deltaY = static_cast<int>(m_yState.output);
    
    return std::make_pair(deltaX, deltaY);
}

