cmake_minimum_required(VERSION 3.13)

project(baidu_cti_cuda_extensions LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

if(CMAKE_VERSION VERSION_LESS 3.18)
  string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES}" 0 1 CTI_ARCH_MAJOR)
  string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES}" 1 1 CTI_ARCH_MINOR)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CTI_ARCH_MAJOR}${CTI_ARCH_MINOR},code=sm_${CTI_ARCH_MAJOR}${CTI_ARCH_MINOR}")
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()

if(NOT DEFINED PYTHON_EXTENSION_SUFFIX)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

if(NOT DEFINED TORCH_CXX11_ABI)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

set(CTI_INSTALL_DIR "${CMAKE_SOURCE_DIR}/cmake_extensions" CACHE PATH "Directory for built Python extension modules")
set(CTI_SMOE_MAXRREGCOUNT "80" CACHE STRING "CUDA max register count for routed SMoE kernels")
set(CTI_ENABLE_CUTLASS_SMOE "0" CACHE STRING "Enable CUTLASS SMoE compile path")
set(CTI_USE_FAST_ATTENTION_EXP "0" CACHE STRING "Use __expf in attention kernels")
set(CTI_CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/third_party/cutlass" CACHE PATH "CUTLASS root")
set(PYTHON_INCLUDE_DIR "" CACHE PATH "Python include directory from sysconfig")

function(add_cti_extension target source)
  add_library(${target} MODULE "${source}")
  target_include_directories(${target} PRIVATE ${TORCH_INCLUDE_DIRS})
  if(PYTHON_INCLUDE_DIR)
    target_include_directories(${target} PRIVATE "${PYTHON_INCLUDE_DIR}")
  endif()
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
  if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(${target} PRIVATE "${TORCH_PYTHON_LIBRARY}")
  endif()
  target_compile_definitions(${target} PRIVATE
    TORCH_EXTENSION_NAME=${target}
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
  )
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(${target} PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions"
  )
  install(TARGETS ${target} LIBRARY DESTINATION "${CTI_INSTALL_DIR}")
endfunction()

add_cti_extension(layernorm_512_ext CUDA/norm_kernels.cu)
add_cti_extension(varlen_attention_ext CUDA/attention_kernels.cu)
add_cti_extension(routed_smoe_ext CUDA/smoe_kernels.cu)
add_cti_extension(embedding_bag_ext CUDA/embedding_bag_kernels.cu)
add_cti_extension(gate_topk_ext CUDA/softmax_topk_kernels.cu)
add_cti_extension(output_ext CUDA/output_kernels.cu)

target_compile_definitions(varlen_attention_ext PRIVATE
  USE_FAST_ATTENTION_EXP=${CTI_USE_FAST_ATTENTION_EXP}
)

target_compile_definitions(routed_smoe_ext PRIVATE
  BAIDU_CTI_ENABLE_CUTLASS_SMOE=${CTI_ENABLE_CUTLASS_SMOE}
)
target_compile_options(routed_smoe_ext PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${CTI_SMOE_MAXRREGCOUNT}>
)

if(CTI_ENABLE_CUTLASS_SMOE)
  set(CTI_CUTLASS_INCLUDE "${CTI_CUTLASS_ROOT}/include")
  set(CTI_CUTLASS_UTIL_INCLUDE "${CTI_CUTLASS_ROOT}/tools/util/include")
  if(NOT EXISTS "${CTI_CUTLASS_INCLUDE}" OR NOT EXISTS "${CTI_CUTLASS_UTIL_INCLUDE}")
    message(FATAL_ERROR "CUTLASS include directories not found under ${CTI_CUTLASS_ROOT}")
  endif()
  target_include_directories(routed_smoe_ext PRIVATE
    "${CTI_CUTLASS_INCLUDE}"
    "${CTI_CUTLASS_UTIL_INCLUDE}"
  )
endif()
