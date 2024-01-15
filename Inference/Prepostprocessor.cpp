#include "Prepostprocessor.h"


Prepostprocessor::Prepostprocessor(Model* model)
    :model_(model)
{
    const auto& shapes = model->inputShapes;
    if (!shapes.empty())
    {
        const auto& shape = shapes[0];
        assert(shape.size() == 4);
        
        blobSize_ = shape.at(3) * shape.at(2) * shape.at(1) * shape.at(0);
        blob_ = new float[blobSize_];
    }

    memInfo_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
}

Prepostprocessor::~Prepostprocessor()
{
    if (blob_)
        delete blob_;

    blob_ = nullptr;
}

std::vector<Ort::Value> Prepostprocessor::Preprocess(const cv::Mat& image)
{
    std::vector<Ort::Value> result;
    cv::Mat resizedImage, floatImage;

    try
    {
        if (!model_ || model_->inputShapes.empty())
            throw std::runtime_error("model_ is nullptr!");

        const auto& inputShape = model_->inputShapes;
        const auto& inputTensorShape = inputShape[0]; // yolov5只有一个 维度输入

        if (!ConvertToRGB(image, resizedImage))
            throw std::runtime_error("failed to convert to rgb!");

        // 归一化为统一大小
        resizedImage = Letterbox(resizedImage, cv::Size(inputTensorShape.at(2), inputTensorShape.at(3)));

        // 映射 0~255到 0~1之间
        resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);

        cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

        // hwc -> chw(height width channels)
        std::vector<cv::Mat> chw(floatImage.channels());
        for (int i = 0; i < floatImage.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob_ + i * floatImageSize.width * floatImageSize.height);
        }
        cv::split(floatImage, chw);

        //std::vector<float>inputTensorValues(blob_, blob_ + blobSize_);

        result.push_back(
            Ort::Value::CreateTensor<float>(memInfo_,
                blob_, blobSize_,
                inputTensorShape.data(), inputTensorShape.size())
        );

    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    return result;
}

std::vector<ResultNode> Prepostprocessor::Postprocess(const std::vector<Ort::Value>& outTensor,
    const cv::Size& originalImageShape,
    float confThreshold, float iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    const auto& Shape = model_->inputShapes[0];
    cv::Size resizedImageShape = { static_cast<int>(Shape[3]), static_cast<int>(Shape[2]) };

    ParseRawOutput(outTensor, confThreshold, boxes, confs, classIds);

    std::vector<int> indices; // store the nms result (index)
    // nms
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    std::vector<ResultNode> detections;

    for (int idx : indices)
    {
        ResultNode det = {};

        GetOriCoords(resizedImageShape, originalImageShape, boxes[idx]);

        det.x = boxes[idx].x;
        det.y = boxes[idx].y;
        det.w = boxes[idx].width;
        det.h = boxes[idx].height;

        det.confidence = confs[idx];
        det.classIdx = classIds[idx];

        detections.emplace_back(det);
    }

    return detections;
}

cv::Mat Prepostprocessor::Letterbox(const cv::Mat& image,
    const cv::Size& newShape,
    const cv::Scalar& color,
    bool scaleFill,
    bool scaleUp,
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


bool Prepostprocessor::ConvertToRGB(const cv::Mat& image, cv::Mat& outImage)
{
    if (image.empty())
        return false;

    int channel = image.channels();
    if (channel == 1)
        cv::cvtColor(image, outImage, cv::COLOR_GRAY2RGB);
    else if (channel == 3)
        cv::cvtColor(image, outImage, cv::COLOR_BGR2RGB);
    else if (channel == 4)
        cv::cvtColor(image, outImage, cv::COLOR_BGRA2RGB);
    else
        return false;

    return true;
}



void Prepostprocessor::GetOriCoords(const cv::Size& currentShape,
    const cv::Size& originalShape, cv::Rect& outCoords)
{
    float gain = std::min(static_cast<float>(currentShape.height) / static_cast<float>(originalShape.height),
        static_cast<float>(currentShape.width) / static_cast<float>(originalShape.width));

    int pad[2] = {
      static_cast<int>((static_cast<float>(currentShape.width) - static_cast<float>(originalShape.width) * gain) / 2.0f),
      static_cast<int>((static_cast<float>(currentShape.height) - static_cast<float>(originalShape.height) * gain) / 2.0f)
    };

    outCoords.x = static_cast<int>(std::round((static_cast<float>(outCoords.x - pad[0]) / gain)));
    outCoords.y = static_cast<int>(std::round((static_cast<float>(outCoords.y - pad[1]) / gain)));

    outCoords.width = static_cast<int>(std::round(((float)outCoords.width / gain)));
    outCoords.height = static_cast<int>(std::round(((float)outCoords.height / gain)));
}


void Prepostprocessor::GetBestClassInfo(std::vector<float>::iterator it,
    const int& numClasses, float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;
    const int otherCnt = 5; // skip x, y, w, h, box_conf

    for (int i = otherCnt; i < numClasses + otherCnt; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - otherCnt;
        }
    }
}

void Prepostprocessor::ParseRawOutput(const std::vector<Ort::Value>& tensor, float conf_threshold, std::vector<cv::Rect>& boxes, std::vector<float>& confs, std::vector<int>& classIds)
{
    auto* rawOutput = tensor.at(0).GetTensorData<float>();
    std::vector<int64_t> outputShape = tensor.at(0).GetTensorTypeAndShapeInfo().GetShape();
    size_t count = tensor.at(0).GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    int numClasses = (int)outputShape.at(2) - YOLOV5_OUTBOX_ELEMENT_COUNT; // 这个受模型影响
    int elementsInBatch = (int)(outputShape.at(1) * outputShape.at(2));

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape.at(2))
    {

        const RawResult* box = reinterpret_cast<const RawResult*>(&(*it));
        float clsConf = box->cls_conf;

        if (clsConf > conf_threshold)
        {
            int centerX = box->cx;
            int centerY = box->cy;
            int width = box->w;
            int height = box->h;
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            GetBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }
}