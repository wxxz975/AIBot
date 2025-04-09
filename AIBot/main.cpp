#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>

#include "WindowsMouseController.h"
#include "DesktopTextureCapturer.h"

void test_mouse_control()
{
	WindowsMouseController controller;
	for (int idx = 0; idx < 100; ++idx) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		controller.MoveMouse(10, 5);
	}
}

void test_screen_capture()
{
	DesktopTextureCapturer capture;

	if (!capture.Initialize()) {
		std::cerr << " failed to init the screen capture!\n";
		return;
	}
	TextureDesc desc{};
	
	int success_count = 0;
	int frame_timeout_count = 0;
	int duplication_failed_count = 0;
	int dxgi_init_failed_count = 0;
	
	std::vector<double> capture_times;
	
	for (int idx = 0; idx < 100; ++idx) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		
		auto start_time = std::chrono::high_resolution_clock::now();
		auto status = capture.CaptureNext(&desc);
		if (status == TextureCaptureStatus::OK) {
			success_count++;
			auto end_time = std::chrono::high_resolution_clock::now();
			double capture_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
			capture_times.push_back(capture_time);
		}
		else if (status == TextureCaptureStatus::FRAME_TIMEOUT) {
			frame_timeout_count++;
			continue;
		}
		else if (status == TextureCaptureStatus::DUPLICATION_FAILED) {
			duplication_failed_count++;
			continue;
		}
		else if (status == TextureCaptureStatus::DXGI_INIT_FAILED) {
			dxgi_init_failed_count++;
			continue;
		}

		if (desc.pTexture) {
			desc.pTexture->Release();
			desc.pTexture = nullptr;
		}
	}
	
	double total_time = std::accumulate(capture_times.begin(), capture_times.end(), 0.0);
	double fps = (success_count * 1000.0) / total_time;
	
	std::cout << "\nCapture Statistics:\n";
	std::cout << "Total attempts: " << 100 << "\n";
	std::cout << "Successful captures: " << success_count << "\n";
	std::cout << "Total capture time: " << total_time << " ms\n";
	std::cout << "Average FPS: " << fps << "\n";
	std::cout << "Frame timeout: " << frame_timeout_count << "\n";
	std::cout << "Duplication failed: " << duplication_failed_count << "\n";
	std::cout << "DXGI init failed: " << dxgi_init_failed_count << "\n";
	
	capture.Close();
}

int main(int argc, char* argv[])
{
	// test mouse control 
	// test_mouse_control();


	// test screen capture
	test_screen_capture();
	return 0;
}