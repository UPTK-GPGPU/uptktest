set(MODULE_ROOT_PATH ${CMAKE_SOURCE_DIR}/src)

message(STATUS "include root path")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
	              ${MODULE_ROOT_PATH}/runtime
#                ${MODULE_ROOT_PATH}/cublas
#                ${MODULE_ROOT_PATH}/cufft
#                ${MODULE_ROOT_PATH}/cufft_runable
                )

include(runtime)
#include(cublas)
#include(cufft)
#include(cufft_runable)

set(CASE_FILES ${MODULE_ROOT_PATH}/main.cpp 
               ${RUNTIME_TEST_FILES}
#               ${CUBLAS_TEST_FILES}
#               ${CUFFT_TEST_FILES}
#               ${CUFFT_RUNABLE_TEST_FILES}
               )
               
message(STATUS "CASE_FILES:${CASE_FILES}")
#add test source file
set(TEST_SOURCE_FILES ${CASE_FILES} PARENT_SCOPE)
