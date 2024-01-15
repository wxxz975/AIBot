#include "ModelParser.h"



Model* ModelParser::parse(Ort::Session* session)
{
    Model* model = new Model();
    if (session == nullptr)
        return nullptr;

    bool valid = parseInput(session, model) && parseOutput(session, model) && parseLabels(session, model);

    return valid ? model : nullptr;
}

bool ModelParser::parseInput(Ort::Session* session, Model* model)
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputCnt;

    if (session == nullptr || model == nullptr)
        return false;

    inputCnt = session->GetInputCount();
    for (size_t idx = 0; idx < inputCnt; ++idx)
    {
        auto name = session->GetInputNameAllocated(idx, allocator);
        model->inputNames.push_back(name.get());
        model->inputNamesPtr.push_back(model->inputNames.back().c_str());

        auto TypeInfo = session->GetInputTypeInfo(idx);
        auto TypeAndShape = TypeInfo.GetTensorTypeAndShapeInfo();

        model->inputShapes.emplace_back(TypeAndShape.GetShape());
    }

    return true;
}

bool ModelParser::parseOutput(Ort::Session* session, Model* model)
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t outputCnt;

    if (session == nullptr || model == nullptr)
        return false;

    outputCnt = session->GetOutputCount();
    for (size_t idx = 0; idx < outputCnt; ++idx)
    {
        auto name = session->GetOutputNameAllocated(idx, allocator);
        model->outputNames.push_back(name.get());
        model->outputNamesPtr.push_back(model->outputNames.back().c_str());

        auto TypeInfo = session->GetOutputTypeInfo(idx);
        auto TypeAndShape = TypeInfo.GetTensorTypeAndShapeInfo();

        model->outputShapes.emplace_back(TypeAndShape.GetShape());
    }

    return true;
}


std::vector<std::string> ModelParser::parseLabelsRaw(const std::string& rawData)
{
    std::vector<std::string> labels;
    std::string data = rawData;
    bool inValue = false;
    size_t start, len;

    for (size_t idx = 0; idx < data.size(); ++idx)
    {
        if (data[idx] != '\'')
            continue;
        // µÈÓÚ '
        if (!inValue)
        {
            start = idx + 1;
            inValue = true;
        }
        else {
            len = idx - start;
            inValue = false;
            labels.push_back(data.substr(start, len));
        }
    }
    return labels;
}

bool ModelParser::parseLabels(Ort::Session* session, Model* model, const std::string& labelKey)
{
    std::string rawJson;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metaData = session->GetModelMetadata();
    auto keys = metaData.GetCustomMetadataMapKeysAllocated(allocator);

    for (const auto& key : keys)
    {
        if (labelKey == key.get())
        {
            auto labelsRaw = metaData.LookupCustomMetadataMapAllocated(labelKey.c_str(), allocator);
            rawJson = std::string(labelsRaw.get());
            model->labels = parseLabelsRaw(rawJson);
            break;
        }
    }

    return !model->labels.empty();
}