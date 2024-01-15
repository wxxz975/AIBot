#pragma once
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "Model.h"


class ModelParser
{
private:
    /* data */
public:
    ModelParser() = default;
    ~ModelParser() = default;

    /// @brief 解析模型中的输入、输出、标签
    /// @param session 以及加载的ort session
    /// @return 返回解析完成后的Model指针
    Model* parse(Ort::Session* session);

private:
    bool parseInput(Ort::Session* session, Model* model);

    bool parseOutput(Ort::Session* session, Model* model);

    /// @brief 解析标签，需要保证这个onnx 中存在这个names的这个键值对, 不然会报错
    /// @param session 已加载的ort session 
    /// @param model 需要输出的model*
    /// @param labelKey 这个标签项再原始数据中的key值
    /// @return 返回是否解析成功
    bool parseLabels(Ort::Session* session, Model* model, const std::string& labelKey = "names");

    std::vector<std::string> parseLabelsRaw(const std::string& rawData);

};
