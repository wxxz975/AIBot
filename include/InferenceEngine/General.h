#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace InferenceEngine
{
    enum class ImageEleType: std::size_t
    {
        u8,
        f32,
        f16,
        i64
    };

    inline std::size_t GetElementSize(ImageEleType type)
    {
        switch (type)
        {
        case ImageEleType::u8:
            return 1;
        case ImageEleType::f32:
            return 4;
        case ImageEleType::f16:
            return 2;
        case ImageEleType::i64:
            return 8;
        default:
            return 0;
        }
    }

    enum class ImageColorFormat: std::size_t
    {
        RGB,
        BGR,
        RGBA,
        BGRA,
        GREY
    };
    template<typename T>
    std::size_t ProductVec(const std::vector<T>& values)
    {
        std::size_t ret = 1;
        for(const auto& v : values)
            ret *= v;
        return ret;
    }

    struct ImageData
    {
        std::string             layout;     // CHW
        ImageEleType            type;       // u8
        ImageColorFormat        format;     // rgb
        std::size_t             channels;   // 3 
        std::size_t             height;     // 640
        std::size_t             width;      // 640
        std::vector<std::uint8_t> data;

        ImageData() {};

        ImageData(const std::string& layout, 
            ImageEleType type, 
            ImageColorFormat format, 
            std::size_t channels, 
            std::size_t height, 
            std::size_t width,
            void* ptr
            ):layout(layout), type(type), 
                format(format), channels(channels), height(height), width(width) 
                {   
                    std::size_t ele_size = GetElementSize(type);
                    std::uint8_t *start = reinterpret_cast<std::uint8_t*>(ptr);
                    std::uint8_t *end = start + channels * height * width * ele_size;
                    data = std::vector<std::uint8_t>(start, end);
                };
            
        ImageData(const std::string& layout, 
            ImageEleType type, 
            ImageColorFormat format, 
            std::size_t channels, 
            std::size_t height, 
            std::size_t width,
            std::vector<std::uint8_t> data
            ):layout(layout), type(type), 
                format(format), channels(channels), height(height), width(width),
                data(data)
                {   
                   
                };

        template<typename T>
        T* as()
        {
            return reinterpret_cast<T*>(data.data());
        }

    };

    typedef std::shared_ptr<ImageData> ImageDataPtr;

    

    struct BoundingBox
    {
        std::int32_t    left;
        std::int32_t    top;
        std::int32_t    width;
        std::int32_t    height;
        float           confidence;
        std::size_t     class_index;

        BoundingBox(){};
        BoundingBox(std::int32_t left, std::int32_t top, 
                std::int32_t width, std::int32_t height, 
                float confidence, std::size_t class_index):
                    left(left), top(top), width(width), height(height),
                    confidence(confidence), class_index(class_index)
                {

                };
    };

    typedef std::vector<BoundingBox> DetectOutput;
    

    struct RawOutput
    {
        ImageEleType                type;
        std::vector<std::size_t>    shape; // output shape
        std::vector<std::uint8_t>   data;

        RawOutput() {};

        RawOutput(ImageEleType type, const std::vector<std::size_t>& shape, void* ptr): 
            type(type), shape(shape)
            {
                std::size_t ele_size = GetElementSize(type);
                std::uint8_t *start = reinterpret_cast<std::uint8_t*>(ptr);
                
                
                std::uint8_t *end = start + ProductVec(shape);
                data = std::vector<std::uint8_t>(start, end);
            };

        RawOutput(ImageEleType type, const std::vector<std::size_t>& shape, const std::vector<std::uint8_t>& data):
            type(type), shape(shape), data(data)
            {

            };

        template<typename T>
        T* as()
        {
            return reinterpret_cast<T*>(data.data());
        }
    };

    typedef std::shared_ptr<RawOutput> RawOutputPtr;

    typedef std::vector<RawOutputPtr> RawOutputPtrVec;

};