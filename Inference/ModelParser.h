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

    /// @brief ����ģ���е����롢�������ǩ
    /// @param session �Լ����ص�ort session
    /// @return ���ؽ�����ɺ��Modelָ��
    Model* parse(Ort::Session* session);

private:
    bool parseInput(Ort::Session* session, Model* model);

    bool parseOutput(Ort::Session* session, Model* model);

    /// @brief ������ǩ����Ҫ��֤���onnx �д������names�������ֵ��, ��Ȼ�ᱨ��
    /// @param session �Ѽ��ص�ort session 
    /// @param model ��Ҫ�����model*
    /// @param labelKey �����ǩ����ԭʼ�����е�keyֵ
    /// @return �����Ƿ�����ɹ�
    bool parseLabels(Ort::Session* session, Model* model, const std::string& labelKey = "names");

    std::vector<std::string> parseLabelsRaw(const std::string& rawData);

};
