#pragma once
#include <vector>
#include <string>


struct Model
{
    Model() = default;
    ~Model() = default;

    std::vector<std::string> labels;

    std::vector<std::vector<int64_t>> inputShapes;

    std::vector<std::vector<int64_t>> outputShapes;

    std::vector<std::string> inputNames;
    std::vector<const char*> inputNamesPtr;

    std::vector<std::string> outputNames;
    std::vector<const char*> outputNamesPtr;
};