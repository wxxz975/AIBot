#pragma once

#include <memory>

/// <summary>
/// Result information used to represent inference output
/// </summary>
struct BoundingBox
{
	float	left;
	float	top;		
	float	width;	
	float	height;
	int		classIndex;
	float	confidence;
};
typedef std::shared_ptr<BoundingBox> PBoundingBox;


struct alignas(float) YoloRawResult
{
    float cx;       // center x
    float cy;       // center y
    float w;        // width
    float h;        // height
    float cls_conf; // class confidence
};


typedef YoloRawResult* pYoloRawResult;

// There are 5 elements in one box
#define YOLOV5_OUTBOX_ELEMENT_COUNT 5