set(CUBLAS_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cublas)

include_directories("utils")
message(STATUS "include cublas api")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
   ${CUBLAS_ROOT_PATH}/extensions
   ${CUBLAS_ROOT_PATH}/level-1
   ${CUBLAS_ROOT_PATH}/level-2
   ${CUBLAS_ROOT_PATH}/level-3

) 


include(extensions)
include(level-1)
include(level-2)
include(level-3)

set(CUBLAS_TEST_FILES 
    ${EXTENSIONS_TEST_FILES}
    ${CUBLAS_LEVEL_1_TEST_FILES}
    ${CUBLAS_LEVEL_2_TEST_FILES}
    ${CUBLAS_LEVEL_3_TEST_FILES}
)
