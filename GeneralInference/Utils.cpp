#include "Utils.h"


namespace GeneralInference
{


	cv::Mat Letterbox(const cv::Mat& image, 
		const cv::Size& newShape, 
		const cv::Scalar& color, 
		bool scaleFill, bool scaleUp, 
		int stride)
	{
        // 参数验证
        if (newShape.width <= 0 || newShape.height <= 0) {
            throw std::invalid_argument("Invalid newShape dimensions");
        }
        cv::Mat outImage;
        cv::Size shape = image.size();
        float scale_factor = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
            static_cast<float>(newShape.width) / static_cast<float>(shape.width));
        if (!scaleUp)
            scale_factor = std::min(scale_factor, 1.0f);

        float width_padding = (newShape.width - shape.width * scale_factor) / 2.f;
        float height_padding = (newShape.height - shape.height * scale_factor) / 2.f;


        // 如果需要填充颜色
        if (scaleFill) {
            width_padding = 0.0f;
            height_padding = 0.0f;
            scale_factor = static_cast<float>(newShape.width) / shape.width;
        }

        // 调整图像大小
        //cv::Mat outImage;
        if (shape.width != static_cast<int>(shape.width * scale_factor) &&
            shape.height != static_cast<int>(shape.height * scale_factor)) {
            cv::resize(image, outImage, cv::Size(static_cast<int>(shape.width * scale_factor),
                static_cast<int>(shape.height * scale_factor)));
        }
        else {
            outImage = image;
        }

        // 计算边界
        int top = static_cast<int>(std::round(height_padding - 0.1f));
        int bottom = static_cast<int>(std::round(height_padding + 0.1f));
        int left = static_cast<int>(std::round(width_padding - 0.1f));
        int right = static_cast<int>(std::round(width_padding + 0.1f));

        // 添加边界
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

        return outImage;
	}
    void Letterbox(const cv::Mat& image, 
        cv::Mat& outImage, 
        const cv::Size& newShape, 
        const cv::Scalar& color, 
        bool auto_, 
        bool scaleFill, bool scaleUp, 
        int stride)
    {
        cv::Size shape = image.size();
        float r = std::min((float)newShape.height / (float)shape.height,
            (float)newShape.width / (float)shape.width);
        if (!scaleUp)
            r = std::min(r, 1.0f);

        float ratio[2]{ r, r };
        int newUnpad[2]{ (int)std::round((float)shape.width * r),
                         (int)std::round((float)shape.height * r) };

        auto dw = (float)(newShape.width - newUnpad[0]);
        auto dh = (float)(newShape.height - newUnpad[1]);

        if (auto_)
        {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            newUnpad[0] = newShape.width;
            newUnpad[1] = newShape.height;
            ratio[0] = (float)newShape.width / (float)shape.width;
            ratio[1] = (float)newShape.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
        {
            cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
        }

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }


    float IoU(const BoundingBox& box1, const BoundingBox& box2)
    {
        float x1 = std::max(box1.left, box2.left);
        float y1 = std::max(box1.top, box2.top);
        float x2 = std::min(box1.left + box1.width, box2.left + box2.width);
        float y2 = std::min(box1.top + box1.height, box2.top + box2.height);

        float intersection_width = std::max(0.0f, x2 - x1);
        float intersection_height = std::max(0.0f, y2 - y1);
        float intersection_area = intersection_width * intersection_height;

        float box1_area = box1.width * box1.height;
        float box2_area = box2.width * box2.height;

        float union_area = box1_area + box2_area - intersection_area;

        return intersection_area / union_area;
    }
   
}