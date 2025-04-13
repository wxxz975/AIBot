#pragma once


#include <cuda_runtime.h>

/// <summary>
/// 二维矩阵wrapper， 主要是用于解析这个矩阵
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T, bool transpose = false>
class Matrix2D {
public:
    // 构造函数，初始化行、列以及数据，默认未转置
    __host__ __device__ Matrix2D(const T* data, size_t rows, size_t cols)
        : m_rows(rows), m_cols(cols), m_data(data) {
    }

    __host__ __device__ const T& operator()(size_t i, size_t j) const {
        if constexpr (transpose) {
            return m_data[j * m_cols + i];
        }
        else {
            return m_data[i * m_cols + j];
        }
    }

    // 返回逻辑行数：若已转置，则行数等于原cols；否则等于原rows
    __host__ __device__ size_t Rows() const { return transpose ? m_cols : m_rows; }
    // 返回逻辑列数：若已转置，则列数等于原rows；否则等于原cols
    __host__ __device__ size_t Cols() const { return transpose ? m_rows : m_cols; }


private:
    size_t m_rows, m_cols;
    const T* m_data;
};

