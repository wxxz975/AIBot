#include "DesktopTextureCapturer.h"

#include <iostream>



bool DesktopTextureCapturer::Initialize(const std::string& windowTitle)
{
    if (!windowTitle.empty()) {
        m_targetWindow = FindWindowA(nullptr, windowTitle.c_str());
        if (!m_targetWindow) {
            return false;
        }
    }

   

    // Initialize DirectX
    HRESULT hr = S_OK;

    // Driver types supported
    D3D_DRIVER_TYPE driverTypes[] = {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    auto numDriverTypes = ARRAYSIZE(driverTypes);

    // Feature levels supported
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_1 };
    auto numFeatureLevels = ARRAYSIZE(featureLevels);

    D3D_FEATURE_LEVEL featureLevel;

    // Create device
    for (size_t i = 0; i < numDriverTypes; i++) {
        hr = D3D11CreateDevice(nullptr, driverTypes[i], nullptr, 0, featureLevels, (UINT)numFeatureLevels,
            D3D11_SDK_VERSION, &m_d3dDevice, &featureLevel, &m_d3dDeviceContext);
        if (SUCCEEDED(hr))
            break;
    }
    if (FAILED(hr))
        return false;

    // Get DXGI device
    IDXGIDevice* dxgiDevice = nullptr;
    hr = m_d3dDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    if (FAILED(hr))
        return false;

    // Get DXGI adapter
    IDXGIAdapter* dxgiAdapter = nullptr;
    hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);
    dxgiDevice->Release();
    dxgiDevice = nullptr;
    if (FAILED(hr)) {
        return false;
    }

    // Get output
    IDXGIOutput* dxgiOutput = nullptr;
    hr = dxgiAdapter->EnumOutputs(0, &dxgiOutput);
    dxgiAdapter->Release();
    dxgiAdapter = nullptr;
    if (FAILED(hr)) {
        return false;
    }

    dxgiOutput->GetDesc(&m_outputDesc);

    if (!InitArgs()) {
        return false;
    }

    // QI for Output 1
    IDXGIOutput1* dxgiOutput1 = nullptr;
    hr = dxgiOutput->QueryInterface(__uuidof(dxgiOutput1), (void**)&dxgiOutput1);
    dxgiOutput->Release();
    dxgiOutput = nullptr;
    if (FAILED(hr))
        return false;

    // Create desktop duplication
    hr = dxgiOutput1->DuplicateOutput(m_d3dDevice, &m_deskDupl);
    dxgiOutput1->Release();
    dxgiOutput1 = nullptr;
    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE) {
            return false;
        }
        return false;
    }

    // Create a persistent GPU texture
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = m_outputDesc.DesktopCoordinates.right - m_outputDesc.DesktopCoordinates.left;
    desc.Height = m_outputDesc.DesktopCoordinates.bottom - m_outputDesc.DesktopCoordinates.top;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    hr = m_d3dDevice->CreateTexture2D(&desc, nullptr, &m_pPersistentTexture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create persistent texture, HRESULT=" << hr << std::endl;
        return false;
    }

    return true;
}

void DesktopTextureCapturer::Close()
{
    if (m_pPersistentTexture) {
        m_pPersistentTexture->Release();
        m_pPersistentTexture = nullptr;
    }

    if (m_deskDupl) {
        m_deskDupl->Release();
        m_deskDupl = nullptr;
    }

    if (m_d3dDeviceContext) {
        m_d3dDeviceContext->Release();
        m_d3dDeviceContext = nullptr;
    }

    if (m_d3dDevice) {
        m_d3dDevice->Release();
        m_d3dDevice = nullptr;
    }
    
    m_haveFrameLock = false;
}

ID3D11Texture2D* DesktopTextureCapturer::GetTheTexture()
{
    return m_pPersistentTexture;
}

const TextureDesc& DesktopTextureCapturer::GetTheTextureDesc()
{
    return m_TextureDesc;
}

TextureCaptureStatus DesktopTextureCapturer::CaptureNext()
{
    if (!m_deskDupl)
        return TextureCaptureStatus::DXGI_INIT_FAILED;

    HRESULT hr;

    if (m_haveFrameLock) {
        m_haveFrameLock = false;
        hr = m_deskDupl->ReleaseFrame();
        if (FAILED(hr)) {
            // 如果释放帧失败，需要重新初始化
            std::cout << " dxgi Acquire failed\n"; // perhaps shutdown and reinitializes
            Close();
            if (!Initialize()) {
                return TextureCaptureStatus::DXGI_INIT_FAILED;
            }
        }
    }

    IDXGIResource* deskRes = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    hr = m_deskDupl->AcquireNextFrame(0, &frameInfo, &deskRes);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        std::cout << "dxgi time out\n";
        return TextureCaptureStatus::FRAME_TIMEOUT;
    }
    if (FAILED(hr)) {
        std::cout << "dxgi Acquire failed, attempting to reinitialize\n";
        // 如果获取帧失败，尝试重新初始化
        Close();
        if (!Initialize()) {
            return TextureCaptureStatus::DXGI_INIT_FAILED;
        }
        return TextureCaptureStatus::DUPLICATION_FAILED;
    }

    m_haveFrameLock = true;

    ID3D11Texture2D* gpuTex = nullptr;
    hr = deskRes->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&gpuTex);
    deskRes->Release();
    deskRes = nullptr;
    if (FAILED(hr)) {
        m_haveFrameLock = false;
        m_deskDupl->ReleaseFrame();
        return TextureCaptureStatus::DUPLICATION_FAILED;
    }

    // Copy the captured texture to the persistent texture
    m_d3dDeviceContext->CopyResource(m_pPersistentTexture, gpuTex);

    gpuTex->Release();

    return TextureCaptureStatus::OK;
}

bool DesktopTextureCapturer::InitArgs()
{
    if (m_targetWindow) {
        RECT windowRect;
        if (!GetClientRect(m_targetWindow, &windowRect)) {
            return false;
        }
        m_rect = windowRect;
    }
    else {
        // 如果未指定窗口，则使用整个屏幕
        m_rect = { 0, 0, m_outputDesc.DesktopCoordinates.right, m_outputDesc.DesktopCoordinates.bottom };
    }
   
    m_TextureDesc.width = m_outputDesc.DesktopCoordinates.right - m_outputDesc.DesktopCoordinates.left;
    m_TextureDesc.height = m_outputDesc.DesktopCoordinates.bottom - m_outputDesc.DesktopCoordinates.top;
    m_TextureDesc.memLayout = TextureMemLayout::HWC;
    m_TextureDesc.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    m_TextureDesc.rowPitch = 4 * m_TextureDesc.width;

    return true;
}

