#include "ModelParser.h"
#include <regex>

namespace GeneralInference
{

	/*

	std::shared_ptr<Model> ModelParser::ParseInput(Ort::Session* session)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		std::size_t inputCnt = 0;
		std::shared_ptr<Model> model = nullptr;
		if (!session)
			return model;

		 
		inputCnt = session->GetInputCount();
		if (inputCnt != 1)
			throw std::runtime_error("Only supports one input!");


		for (size_t idx = 0; idx < inputCnt; ++idx)
		{
			auto name = session->GetInputNameAllocated(idx, allocator);
			
			model->inputNames.push_back(name.get());
			model->inputNamesPtr.push_back(model->inputNames.back().c_str());

			auto TypeInfo = session->GetInputTypeInfo(idx);
			auto TypeAndShape = TypeInfo.GetTensorTypeAndShapeInfo();

			model->inputShapes.emplace_back(TypeAndShape.GetShape());
		}
	}*/

	

	std::vector<std::string> ModelParser::ParseLabels(Ort::Session* session, const std::string& label_key)
	{
		std::string raw_json;
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::ModelMetadata metaData = session->GetModelMetadata();
		auto keys = metaData.GetCustomMetadataMapKeysAllocated(allocator);
		std::vector<std::string> labels;

		for (const auto& key : keys)
		{
			if (label_key != key.get())
				continue;

			auto labelsRaw = metaData.LookupCustomMetadataMapAllocated(label_key.c_str(), allocator);
			raw_json = std::string(labelsRaw.get());
			
			labels = ParseLabelsRaw(raw_json);
			break;
		}

		
		return labels;
	}

	std::vector<std::string> ModelParser::ParseLabelsRaw(const std::string& raw_json)
	{
		std::vector<std::string> labels;

		// 正则表达式模式
		std::regex pattern("'([^']*)'");

		// 迭代器对正则表达式进行匹配
		std::sregex_iterator it(raw_json.begin(), raw_json.end(), pattern);
		std::sregex_iterator end;

		while (it != end) {
			std::smatch match = *it;
			std::string value = match[1].str();
			labels.push_back(value);
			++it;
		}

		return labels;
	}

	

}