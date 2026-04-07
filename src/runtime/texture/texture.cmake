message(STATUS "include texture test case")
set(RUNTIME_TEXTURE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/texture)

set(RUNTIME_TEXTURE_TEST_FILES
    ${RUNTIME_TEXTURE_DIR}/cudaBindTex2DPitch.cu
    ${RUNTIME_TEXTURE_DIR}/cudaBindTexRef1DFetch.cu
    ${RUNTIME_TEXTURE_DIR}/cudaCreateTextureObject_ArgValidation.cu
    ${RUNTIME_TEXTURE_DIR}/cudaCreateTextureObject_Array.cu
    ${RUNTIME_TEXTURE_DIR}/cudaCreateTextureObject_Linear.cu
    ${RUNTIME_TEXTURE_DIR}/cudaCreateTextureObject_Pitch2D.cu
    ${RUNTIME_TEXTURE_DIR}/cudaGetChanDesc.cu
    # ${RUNTIME_TEXTURE_DIR}/cudaNormalizedFloatValueTex.cu
    ${RUNTIME_TEXTURE_DIR}/cudaSimpleTexture2DLayered.cu
    ${RUNTIME_TEXTURE_DIR}/cudaSimpleTexture3D.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTex1DFetchCheckModes.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTexObjPitch.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureMipmapObj2D.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureObj1DCheckModes.cu
    # ${RUNTIME_TEXTURE_DIR}/cudaTextureObj1DCheckSRGBModes.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureObj1DFetch.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureObj2D.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureObj2DCheckModes.cu
    # ${RUNTIME_TEXTURE_DIR}/cudaTextureObj2DCheckSRGBModes.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureObj3DCheckModes.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureObjFetchVector.cu
    ${RUNTIME_TEXTURE_DIR}/cudaTextureRef2D.cu
    ${RUNTIME_TEXTURE_TEST_TMP_FILES}
)

