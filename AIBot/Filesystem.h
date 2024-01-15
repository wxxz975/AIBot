#pragma once
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;
typedef std::string Path;


namespace IFilesystem
{

    /// <summary>
    /// ��ȡ���argv0 �����ִ���ļ����ڵ��ļ���
    /// </summary>
    /// <param name="argv0">���main�д����argv[0]</param>
    /// <returns></returns>
    Path GetExecutablePath(const std::string& argv0);

    /// @brief ��ȡ��ǰ��ִ���ļ���·��
    /// @return �������·��
    Path GetCurrentPath();

    /// @brief ����һ���ļ���
    /// @param path ��Ҫ�������ļ�·��
    /// @return �����Ƿ񴴽��ɹ�
    bool CreateDir(const Path& path);

    /// @brief ���Ȼ�ȡ��ǰ�Ŀ�ִ���ļ�·�������Ұ������������·��reletive_pathƴ�ӵ���ִ��·������
    /// @param relative_path ���·��
    /// @return ����ƴ�Ӻõ�·��
    Path JoinCurrentPath(const Path& relative_path);

    /// @brief ƴ������·��������ظ����� / \\������
    /// @param left ��ߵ�·��
    /// @param right �ұߵ�·��
    /// @return ����ƴ�Ӻõ�·��
    Path ConcatPath(const Path& left, const Path& right);


    /// @brief ��ȡһ��·���е��ļ���
    /// @param path ����·��
    /// @return �����ļ���
    std::string GetFilename(const Path& path);

    /// @brief ��ȡһ�������ļ�·���еĺ�׺
    /// @param path ·��
    /// @return ���غ�׺
    std::string GetExtension(const Path& path);

    /// @brief ��ȡ����Ŀ¼
    /// @param path ������·��
    /// @return ���ظ���Ŀ¼
    Path GetParentPath(const Path& path);


    /// @brief �����ļ��ĺ�׺ƥ��
    /// @param path ��ѯ��·��
    /// @param ext ��׺, ������, �����׺ onnx, jpg
    /// @return ���ػ�ȡ����·������
    std::vector<std::string> GetFilesBySuffix(const Path& path, const std::string& ext);


    /// @brief ����������·����ȡ���������ʾ��·��
    /// @param path ��Ҫ��ѯ��·��
    /// @param reg ������ʽ
    /// @return ���ػ�ȡ����·������
    std::vector<std::string> GetFilesByRegex(const Path& path, const std::string reg);

    /// @brief �ж�ĳ���ļ�����·���Ƿ����
    /// @param path ·��
    /// @return true ����
    bool IsExist(const Path& path);

    /// @brief �ж��Ƿ����ļ���
    /// @param path 
    /// @return 
    bool IsDir(const Path& path);

    /// @brief �ж��Ƿ����ļ�
    /// @param path 
    /// @return 
    bool IsFile(const Path& path);

    /// @brief �ж����string�Ƿ�����ending��β
    /// @param string 
    /// @param end 
    /// @return 
    bool IsEndWith(const std::string& string, const std::string& ending);


    /// @brief �ж����string�Ƿ�����prefix��ʼ
    /// @param string 
    /// @param prefix 
    /// @return 
    bool IsStartWith(const std::string& string, const std::string& prefix);


    /// <summary>
    /// ʹ��������ʽ��string��ƥ������pattern�鿴�Ƿ�ƥ��ɹ�
    /// </summary>
    /// <param name="string">ԭʼ�ַ���</param>
    /// <param name="pattern">ģʽ</param>
    /// <returns></returns>
    bool RegexMatch(const std::string& string, const std::string& pattern);
};