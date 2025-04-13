#pragma once

#include <d3d11.h>
#include <cuda_runtime.h>



struct TextureDesc;
struct cudaGraphicsResource;


class TextureMapper {
public:
    // Constructor
    TextureMapper();

    // Destructor to ensure resources are released
    ~TextureMapper();

    bool BindResource(ID3D11Texture2D* pPersistenTexture);

    // update texture
    cudaArray_t UpdateTextureToCuda();

    // Unmap texture
    void Destory();

private:
    cudaGraphicsResource*   m_cudaResource;           // CUDA graphics resource
    ID3D11Texture2D*        m_pPersistentTexture = nullptr;
    cudaArray_t             m_mappedArray;            // Mapped CUDA array
};


