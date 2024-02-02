#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace GeneralInference
{

	/// <summary>
	/// Store input and output names and dimension information
	/// </summary>
	struct ModelIOInfo
	{
		std::vector<std::string> names;
		std::vector<const char*> names_ptr;
		std::vector<std::vector<int64_t>> shapes;
		
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
	};

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