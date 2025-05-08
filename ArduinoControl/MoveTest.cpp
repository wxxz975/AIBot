#include <windows.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

using namespace std;

class SerialPort {
public:
    SerialPort(const string& portName) : m_portName(portName), m_hSerial(INVALID_HANDLE_VALUE) {
        openSerialPort();
    }

    ~SerialPort() {
        closeSerialPort();
    }

    void writeToSerialPort(const string& data) {
        DWORD bytesWritten;
        if (!WriteFile(m_hSerial, data.c_str(), data.size(), &bytesWritten, NULL)) {
            cerr << "Error writing to serial port" << endl;
        }
    }

private:
    string m_portName;
    HANDLE m_hSerial;

    void openSerialPort() {
        m_hSerial = CreateFile(m_portName.c_str(),
                               GENERIC_READ | GENERIC_WRITE,
                               0,
                               0,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL,
                               0);

        if (m_hSerial == INVALID_HANDLE_VALUE) {
            cerr << "Error opening serial port" << endl;
            exit(1);
        }

        DCB dcbSerialParams = {0};
        dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

        if (!GetCommState(m_hSerial, &dcbSerialParams)) {
            cerr << "Error getting state" << endl;
            CloseHandle(m_hSerial);
            exit(1);
        }

        dcbSerialParams.BaudRate = CBR_9600;
        dcbSerialParams.ByteSize = 8;
        dcbSerialParams.StopBits = ONESTOPBIT;
        dcbSerialParams.Parity = NOPARITY;

        if (!SetCommState(m_hSerial, &dcbSerialParams)) {
            cerr << "Error setting state" << endl;
            CloseHandle(m_hSerial);
            exit(1);
        }
    }

    void closeSerialPort() {
        if (m_hSerial != INVALID_HANDLE_VALUE) {
            CloseHandle(m_hSerial);
        }
    }
};

int main() {
    string portName = "\\\\.\\COM3"; // 请根据实际情况修改
    SerialPort serialPort(portName);

    while (true) {
        string moveX = "10";
        string moveY = "10";

        cout << "Moving X coord: " << moveX << endl;
        cout << "Moving Y coord: " << moveY << endl;

        this_thread::sleep_for(chrono::milliseconds(100));

        string command = moveX + ":" + moveY + "x";
        serialPort.writeToSerialPort(command);

        cout << "Done. To quit enter q" << endl;
    }

    return 0;
}
