#pragma once


typedef struct _DetectionResultNode
{
    float x, y, w, h;
    int classIdx;
    float confidence;
}ResultNode, * pResultNode;


#ifdef _WIN32
struct alignas(float) RawResult
{
    float cx;       // center x
    float cy;       // center y
    float w;        // width
    float h;        // height
    float cls_conf; // class confidence
};
#else
struct RawResult
{
    float cx;       // center x
    float cy;       // center y
    float w;        // width
    float h;        // height
    float cls_conf; // class confidence
}__attribute__((packed));
#endif

typedef RawResult* pRawResult;

// 占用多少个， 一个box中占用5个元素
#define YOLOV5_OUTBOX_ELEMENT_COUNT 5