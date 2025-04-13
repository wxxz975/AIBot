#include "TextureMapper.h"


#include <iostream>

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

#include "DesktopTextureCapturer.h"
#include "CudaDebug.hpp"



TextureMapper::TextureMapper()
    :m_cudaResource(nullptr), m_mappedArray(nullptr)
{
    
}

TextureMapper::~TextureMapper()
{
    
    Destory();
}

bool TextureMapper::BindResource(ID3D11Texture2D* pPersistenTexture)
{
    m_pPersistentTexture = pPersistenTexture;
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_pPersistentTexture, cudaGraphicsRegisterFlagsNone));
    return true;
}

cudaArray_t TextureMapper::UpdateTextureToCuda()
{
    // Map the resource
    cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaResource, 0));
   
    // Get the mapped array
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&m_mappedArray, m_cudaResource, 0, 0));

    return m_mappedArray;
}


void TextureMapper::Destory()
{
    if (m_cudaResource) {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaResource, 0));
        CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaResource));
        m_cudaResource = nullptr;
        m_mappedArray = nullptr;
    }
}

