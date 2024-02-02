#include <iostream>
#include <regex>



int main(int argc, char* argv[])
{
    std::string data = "{0: 'Gun', 1: 'Knife', 2: 'Pliers', 3: 'Scissors', 4: 'Wrench'}";

    // ������ʽģʽ
    std::regex pattern("'([^']*)'");

    // ��������������ʽ����ƥ��
    std::sregex_iterator it(data.begin(), data.end(), pattern);
    std::sregex_iterator end;

    while (it != end) {
        std::smatch match = *it;
        std::string value = match[1].str();
        std::cout << "Value: " << value << std::endl;
        ++it;
    }


    return 0;

}