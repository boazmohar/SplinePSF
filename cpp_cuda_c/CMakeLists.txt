cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(SplinePSF LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDA)

if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)  # the latter is a user definable variable
   enable_language(CUDA)
   add_compile_definitions(CUDA_ENABLED)

   if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 90)
   endif()

else()  # NO CUDA
   message(WARNING "No CUDA compiler found or disabled by user defined command line variable '-DCUDA_DISABLED=True'.
   You may need to specify its path according to the CMake instructions applicable to this version.")

endif()


add_library(spline_psf_cpu_impl STATIC src/spline_psf.c)
target_include_directories(spline_psf_cpu_impl PUBLIC include)
target_compile_features(spline_psf_cpu_impl PUBLIC c_std_99)
set_property(TARGET spline_psf_cpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)
   add_library(spline_psf_cu_impl STATIC src/spline_psf_gpu.cu)
   target_include_directories(spline_psf_cu_impl PUBLIC include)
   target_compile_features(spline_psf_cu_impl PUBLIC cxx_std_11)
   # CUDA_SEPARABLE_COMPILATION ON is REQUIRED here: spline_psf_gpu.cu uses
   # dynamic parallelism (kernels launched from device code in fAt3DVolumeROI
   # and kernel_derivative_ROI), which only works in separable mode.
   set_target_properties(
      spline_psf_cu_impl
      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
      )

   # PRISM/Janelia HPC fix: use the CUDA version CMake detected (matches PyTorch),
   # not the /usr/local/cuda symlink which may point to a newer toolkit and cause
   # "nvlink fatal: Input file ... newer than toolkit" errors.
   link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
   include_directories("${CUDA_INCLUDE_DIRS}")

   # PRISM fix: NO_EXTRAS disables pybind11's automatic -flto=auto, which
   # otherwise makes the final .so lose its local `fatbinData@Base` symbol
   # and fail to dlopen with `undefined symbol: fatbinData`. The LTO flag
   # is not overridable via INTERPROCEDURAL_OPTIMIZATION because pybind11
   # adds it through its own interface library (pybind11::thin_lto).
   pybind11_add_module(spline NO_EXTRAS src/pybind_spline.cpp)

   # PRISM fix: the pybind module links against a CUDA static library with
   # CUDA_SEPARABLE_COMPILATION ON. These settings force the device-link step
   # on this target and make nvcc the final linker so libcudart/libcudadevrt
   # get pulled in automatically.
   set_target_properties(spline PROPERTIES
      LINKER_LANGUAGE CUDA
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
   target_link_libraries(spline PRIVATE spline_psf_cu_impl spline_psf_cpu_impl)

else()  # NO CUDA
   pybind11_add_module(spline src/pybind_spline.cpp)
   target_link_libraries(spline PRIVATE spline_psf_cpu_impl)

endif()
