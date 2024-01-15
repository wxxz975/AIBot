#include "InferenceEngine.h"
#include "ModelParser.h"
#include "Misc.h"
#include "Filesystem.h"



InferenceEngine::InferenceEngine()
{
}

InferenceEngine::~InferenceEngine()
{
    if (model_){
        delete model_;
        model_ = nullptr;
    }

    if (processor_) {
        delete processor_;
        processor_ = nullptr;
    }
}

bool InferenceEngine::Initialize(const std::string& model_path)
{
	return CreateSession(model_path) && ParseModel() && WarmUpModel();
}



std::vector<ResultNode> InferenceEngine::Infer(const cv::Mat& img)
{
    std::vector<ResultNode> result;
    const auto& inputNames = model_->inputNamesPtr;
    const auto& outputNames = model_->outputNamesPtr;

    if (processor_)
    {
        auto inputTensor = processor_->Preprocess(img);

        std::vector<Ort::Value>outTensor = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensor.data(), inputTensor.size(), outputNames.data(), outputNames.size());

        result = processor_->Postprocess(outTensor, img.size(), confidenceThreshold_, iouThreshold_);
    }

    return result;
}

bool InferenceEngine::CreateSession(const std::string& modelPath)
{
    if (!IFilesystem::IsExist(modelPath))
        return false;

    envName_ = IFilesystem::GetFilename(modelPath);
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, envName_.c_str());

    sessionOpt = Ort::SessionOptions();
    sessionOpt.SetIntraOpNumThreads(0);
    sessionOpt.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    //sessionOpt.DisableCpuMemArena();

    isGpuAval = IsGPUAvailable();

    if (isGpuAval)
    {
        
        auto cudaOptions = CreateCudaOptions();

        std::string test = "Append before!\n";
        try
        {
            sessionOpt.AppendExecutionProvider_CUDA(cudaOptions);
        }
        catch (const std::exception& exp)
        {
            std::cout << "exp:" << exp.what() << "\n";
        }
        
    }

    std::wstring wmodelPath = std::wstring(modelPath.begin(), modelPath.end());

    session_ = Ort::Session(env_, wmodelPath.c_str(), sessionOpt);
    return true;
}

bool InferenceEngine::ParseModel()
{
    model_ = ModelParser().parse(&session_);
    if (!model_)
        return false;
    processor_ = new Prepostprocessor(model_);

    return true;
}

OrtCUDAProviderOptions InferenceEngine::CreateCudaOptions()
{
    OrtCUDAProviderOptions cudaOption;
    cudaOption.device_id = 0;
    cudaOption.arena_extend_strategy = 0;
    cudaOption.gpu_mem_limit = std::numeric_limits<size_t>::max();
    cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cudaOption.do_copy_in_default_stream = 1;

    return cudaOption;
}

bool InferenceEngine::IsGPUAvailable()
{
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(),
        "CUDAExecutionProvider");

    return cudaAvailable != availableProviders.end();
}

cv::Mat InferenceEngine::Render(const cv::Mat& img, const std::vector<ResultNode>& results)
{
    return RenderBoundingBoxes(img, results);
}
