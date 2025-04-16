#include <cuda_runtime.h>

/*
struct DetectionBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    float classId;
    float suppressed;

    DetectionBox() : x1(0), y1(0), x2(0), y2(0), confidence(0), classId(0), suppressed(0) {};

    __host__ __device__ float Area() const {
        return (x2 - x1 + 1.0f) * (y2 - y1 + 1.0f);
    }

    __host__ __device__ float IoU(const DetectionBox& other)
    {
        float inter = __max(0.0f, __min(x2, other.x2) - __max(x1, other.x1) + 1.0f) *
            __max(0.0f, __min(y2, other.y2) - __max(y1, other.y1) + 1.0f);
        float unionArea = Area() + other.Area() - inter;
        return unionArea > 0 ? inter / unionArea : 0;
    }
};
#define RAW_YOLOV8_OUTPUTS_ELEMENT (sizeof(DetectionBox) / sizeof(float))


#define YOLOV8_MAX_NUM_OBJ 1000



struct RawYOLOv8Outputs
{
    float center_x;
    float center_y;
    float width;
    float height;
    // ... 
    // class confidence
};

#define YOLOV8OutputElement (sizeof(RawYOLOv8Outputs) / sizeof(float))
*/