#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "WindowsMouseController.h"
#include "DesktopTextureCapturer.h"
#include "TextureMapper.h"

#include "CudaDebug.hpp"

void test_mouse_control()
{
	WindowsMouseController controller;
	for (int idx = 0; idx < 100; ++idx) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		controller.MoveMouse(10, 5);
	}
}

/*
	目前可以实现将这个截取到的数据通过映射到cuda内存，fps基本上300+
*/
void test_screen_capture()
{
	DesktopTextureCapturer capture;
	TextureMapper mapper;

	if (!capture.Initialize()) {
		std::cerr << " failed to init the screen capture!\n";
		return;
	}
	if (!mapper.BindResource(capture.GetTheTexture())) {
		std::cerr << "Failed to bind the texture resource to cuda!\n";
		return;
	}


	auto desc = capture.GetTheTextureDesc();
	void* d_ptr = nullptr, *h_ptr = nullptr;
	size_t pitch = 0, bytes = 0;
	bytes = desc.width * sizeof(uchar4) * desc.height;
	CUDA_CHECK(cudaMallocPitch(&d_ptr, &pitch, desc.width * sizeof(uchar4), desc.height));
	

	int success_count = 0;
	int frame_timeout_count = 0;
	int duplication_failed_count = 0;
	int dxgi_init_failed_count = 0;
	
	std::vector<double> capture_times;
	cv::Mat hostImage = cv::Mat(desc.height, desc.width, CV_8UC4);
	
	int idx = 0;
	while (true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		idx++;
		if (cv::waitKey(1) == 27) {
			break;
		}

		auto start_time = std::chrono::high_resolution_clock::now();
		auto status = capture.CaptureNext();

		if (status == TextureCaptureStatus::OK) {
			success_count++;
			auto array = mapper.UpdateTextureToCuda();

			CUDA_CHECK(cudaMemcpy2DFromArray(d_ptr, pitch, array, 0, 0, desc.width * sizeof(uchar4), desc.height, cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy2D(
				hostImage.data,             // 主机内存指针
				hostImage.step,             // 每行字节数（Mat自动计算对齐）
				d_ptr,                   // 设备端线性内存数据
				desc.width * sizeof(uchar4),                // 设备端每行字节数
				desc.width * sizeof(uchar4),     // 每行拷贝的字节数
				desc.height,                     // 行数
				cudaMemcpyDeviceToHost      // 设备到主机拷贝
			));

			cv::Mat displayImage;
			cv::cvtColor(hostImage, displayImage, cv::COLOR_BGRA2BGR);

			cv::imshow("CUDA Captured Image", displayImage);

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
	}
	
	double total_time = std::accumulate(capture_times.begin(), capture_times.end(), 0.0);
	double fps = (success_count * 1000.0) / total_time;
	
	std::cout << "\nCapture Statistics:\n";
	std::cout << "Total attempts: " << idx << "\n";
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