#include "Filesystem.h"
#include <regex>
#include <fstream>
#include <sstream>

namespace IFilesystem
{
    Path GetCurrentPath()
    {
        return fs::current_path().string();
    }

    bool CreateDir(const Path& path)
    {
        return fs::create_directory(path);;
    }

    Path JoinCurrentPath(const Path& relative_path)
    {
        Path curPath = GetCurrentPath();
        return ConcatPath(curPath, relative_path);
    }

    Path ConcatPath(const Path& left, const Path& right)
    {
        fs::path lpath(left);
        fs::path rpath(right);

        return (lpath / rpath).string();
    }

    std::string GetFilename(const Path& path)
    {
        return fs::path(path).filename().string();
    }

    std::string GetExtension(const Path& path)
    {
        return fs::path(path).extension().string();
    }

    Path GetParentPath(const Path& path)
    {
        return fs::path(path).parent_path().string();
    }


    ///±ÈÈçÆ¥Åäjpgºó×ºµÄÎÄ¼þ "\\.jpg$"
    std::vector<std::string> GetFilesBySuffix(const Path& path, const std::string& extPat)
    {
        std::vector<std::string> result;
        if (IsExist(path)) {
            for (const auto& entry : std::filesystem::directory_iterator(path))
            {
                if (!entry.is_regular_file())
                    continue;

                std::string extension = entry.path().extension().string();
                if (extension.size() < 1)
                    continue;

                extension = extension.substr(1);

                if (extension != extPat)
                    continue;

                result.push_back(entry.path().string());
            }
        }

        return result;
    }

    std::vector<std::string> GetFilesByRegex(const Path& path, const std::string reg)
    {
        std::vector<std::string> result;
        std::regex regex(reg);
        if (IsExist(path)) {
            for (const auto& entry : std::filesystem::directory_iterator(path))
            {
                if (!entry.is_regular_file())
                    continue;

                std::string filename = entry.path().filename().string();
                if (!RegexMatch(filename, reg))
                    continue;

                result.push_back(entry.path().string());
            }
        }

        return result;
    }

    bool IsExist(const Path& path)
    {
        return fs::exists(path);
    }


    bool IsDir(const Path& path)
    {
        return fs::is_directory(path);
    }

    bool IsFile(const Path& path)
    {
        return !IsDir(path);
    }

    bool IsEndWith(const std::string& string, const std::string& ending)
    {
        if (string.length() < ending.length()) {
            return false;
        }
        return string.substr(string.length() - ending.length()) == ending;
    }

    bool IsStartWith(const std::string& string, const std::string& prefix)
    {
        if (prefix.length() > string.length())
            return false;

        return string.compare(0, prefix.length(), prefix) == 0;
    }

    bool RegexMatch(const std::string& string, const std::string& pattern)
    {
        std::regex pat(pattern);
        return std::regex_match(string, pat);
    }
    bool ReadFile(const std::string& filepath, std::vector<std::uint8_t>& outdata)
    {
        if (!IsExist(filepath)) return false;
        std::ifstream fp(filepath, std::ios::in | std::ios::binary);

        fp.seekg(0, std::ios::end);
        std::streamsize filesize = fp.tellg();
        fp.seekg(0, std::ios::beg);

        outdata.resize(filesize);

        fp.read(reinterpret_cast<char*>(outdata.data()), filesize);

        return true;
    }

    bool WriteFile(const std::string& filepath, const std::vector<std::uint8_t>& data)
    {
        if (data.empty()) return true;

        std::ofstream fp(filepath, std::ios::out | std::ios::binary);

        fp.write(reinterpret_cast<const char*>(data.data()), data.size());

        return true;
    }

    bool DeleteFile(const std::string& filepath)
    {
        return fs::remove(filepath);
    }


    std::string GetSuffix(const Path& filepath)
    {
        size_t lastDotPos = filepath.find_last_of(".");
        if (lastDotPos == std::string::npos) return ""; // 没有点号

        size_t lastSlashPos = filepath.find_last_of("/\\");
        // 如果点号在最后一个路径分隔符之前，说明没有扩展名
        if (lastSlashPos != std::string::npos && lastSlashPos > lastDotPos) return "";

        return filepath.substr(lastDotPos + 1);
    }

    bool Compare(const std::string& str1, const std::string& str2, bool caseSensitive)
    {
        if (str1.size() != str2.size()) {
            return false;
        }

        if (caseSensitive) {
            return str1 == str2;
        }
        else {
            // 大小写不敏感比较
            return std::equal(str1.begin(), str1.end(), str2.begin(), str2.end(),
                [](char c1, char c2) {
                    return std::tolower(static_cast<unsigned char>(c1)) ==
                        std::tolower(static_cast<unsigned char>(c2));
                });
        }
    }

    std::vector<std::string> ListDir(const std::string& path)
    {
        std::vector<std::string> entries;

        try {
            // 遍历目录内容
            for (const auto& entry : fs::directory_iterator(path)) {
                entries.push_back(entry.path().string());  // 获取完整路径并添加到 vector 中
            }
        }
        catch (const fs::filesystem_error& e) {
            //std::cerr << "Error accessing path: " << e.what() << '\n';
        }

        return entries;
    }
};