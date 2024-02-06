#pragma once

#include <memory>
#include <string>
#include <vector>

/// <summary>
/// Result information used to represent inference output
/// </summary>
struct BoundingBox
{
	std::int32_t left;
	std::int32_t top;		
	std::int32_t width;
	std::int32_t height;
	std::size_t	 classIndex;
	std::float_t confidence;
	BoundingBox(std::int32_t x, std::int32_t y, 
		std::int32_t w, std::int32_t h, 
		std::size_t idx, std::float_t conf)
		:left(x), top(y), width(w), height(h), classIndex(idx), confidence(conf)
	{};

};
typedef std::shared_ptr<BoundingBox> PBoundingBox;


struct alignas(float) Yolov5RawResult
{
    float cx;       // center x
    float cy;       // center y
    float w;        // width
    float h;        // height
    float box_conf; // box confidence
	// class confidence
};
typedef Yolov5RawResult* pYolov5RawResult;


struct alignas(float) Yolov8RawResult
{
	float cx;	// center x
	float cy;	// center y
	float w;	// width
	float h;	// height
	// class confidence
};



// There are 5 elements in one box, include cx, cy, width, height, box_conf, other is class confidence
#define YOLOV5_OUTBOX_ELEMENT_COUNT sizeof(Yolov5RawResult)/sizeof(float)

//#define YOLOV8_OUTBOX_ELEMENT_COUNT sizeof(Yolov8RawResult)/sizeof(std::int32_t)
#define YOLOV8_OUTBOX_ELEMENT_COUNT sizeof(Yolov8RawResult)/sizeof(float)

/// <summary>
	/// Store input and output names and dimension information
	/// </summary>
struct ModelIOInfo
{
	std::vector<std::string> names;
	std::vector<const char*> names_ptr;
	std::vector<std::vector<int64_t>> shapes;

	ModelIOInfo() {};

	ModelIOInfo(const std::vector<std::string>& i_names,
		const std::vector<const char*> i_names_ptr,
		const std::vector<std::vector<int64_t>>& i_shapes)
		: names(i_names), names_ptr(i_names_ptr), shapes(i_shapes)
	{

	}

	inline void Append(const std::string& name, const std::vector<int64_t>& shape) {
		names.push_back(name);
		names_ptr.push_back(names.back().c_str());
		shapes.push_back(shape);
	}
};

struct Model
{
	ModelIOInfo input, output;
	std::vector<std::string> labels;
	
	Model() {};
};
