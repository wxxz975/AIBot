#pragma once
#include <Windows.h>
#include <string>
#include <d3d11.h>
#include <dxgi1_2.h>


enum class TextureCaptureStatus {
	OK = 0,
	DXGI_INIT_FAILED,
	DUPLICATION_FAILED,
	GPU_RESOURCE_LOST,
	FRAME_TIMEOUT,
	ACCESS_DENIED
};

enum class TextureMemLayout {
	CHW, 
	HWC
};

typedef struct TextureDesc {
	DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
	int width = 0;
	int height = 0;
	int rowPitch = 0;
	TextureMemLayout memLayout = TextureMemLayout::HWC;
}TextureDesc, *pTextureDesc;


/**
 * @brief 桌面纹理捕获器
 * 
 * 该类负责捕获桌面上的纹理，并将其转换为D3D11纹理。
 * 需要手动释放TextureDesc中的pTexture， 否则会造成内存泄漏， 并且释放这个单独用一个线程是效果最好的
 */
class DesktopTextureCapturer
{
public:
	DesktopTextureCapturer() = default;
	~DesktopTextureCapturer() = default;

	bool Initialize(const std::string& title = "");
	void  Close();

	ID3D11Texture2D* GetTheTexture();
	const TextureDesc& GetTheTextureDesc();
	
	TextureCaptureStatus CaptureNext();


private:
	bool InitArgs();

private:
	ID3D11Device* m_d3dDevice = nullptr;
	ID3D11DeviceContext* m_d3dDeviceContext = nullptr;
	IDXGIOutputDuplication* m_deskDupl = nullptr;
	DXGI_OUTPUT_DESC m_outputDesc = {};
	bool m_haveFrameLock = false;

	RECT m_rect = { 0 };
	HWND m_targetWindow = nullptr;

	ID3D11Texture2D* m_pPersistentTexture = nullptr;

	TextureDesc m_TextureDesc;
};

