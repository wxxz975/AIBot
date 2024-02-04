#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace GeneralInference
{

	

	class ModelParser
	{
	public:
		ModelParser() = delete;
		~ModelParser() = delete;

		/*
		static std::shared_ptr<Model> ParseInput(Ort::Session* session);
		static std::shared_ptr<Model> ParseOutput(Ort::Session* session);
		*/

		/// <summary>
		/// 
		/// </summary>
		/// <param name="session"></param>
		/// <param name="label_key"></param>
		/// <returns></returns>
		static std::vector<std::string> ParseLabels(Ort::Session* session, const std::string& label_key = "names");

		static std::vector<std::string> ParseLabelsRaw(const std::string& raw_json);
	};

	

}