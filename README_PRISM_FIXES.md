# SplinePSF — PRISM / Janelia HPC Fixes

This fork of [Haydnspass/SplinePSF](https://github.com/Haydnspass/SplinePSF)
adds Janelia HPC compatibility fixes for the CUDA build on top of the
upstream `master` branch. All fixes live in `cpp_cuda_c/CMakeLists.txt`.

## Fixes Applied (branch `janelia-hpc-fixes`)

### Fix A — CUDA Library Path: Hardcoded → Variable

**Problem:** Without this fix, CMake links against `/usr/local/cuda/lib64`,
which on Janelia HPC is a symlink to the system default CUDA (e.g. 12.9).
But the build uses a specific toolkit version (e.g. 12.1, matched to PyTorch).
This causes an `nvlink` version mismatch at link time:

```
nvlink fatal : Input file '...' newer than toolkit (129 vs 121) (target: sm_50)
```

**Fix:** Added `link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)` and
`include_directories("${CUDA_INCLUDE_DIRS}")` inside the CUDA branch,
right after the `set_target_properties(...)` block. This uses the CUDA
version detected by CMake's `find_package(CUDA)` instead of the system
symlink.

### Fix B — `undefined symbol: fatbinData` at import

**Problem:** After Fix A, the build succeeds but `import spline` crashes with:

```
ImportError: .../spline.cpython-311-x86_64-linux-gnu.so: undefined symbol: fatbinData
```

**Root cause:** The pybind module (`spline`) is a pure-C++ target that links
against the CUDA static library `spline_psf_cu_impl` built with
`CUDA_SEPARABLE_COMPILATION ON` (required — `spline_psf_gpu.cu` uses dynamic
parallelism: kernels launch kernels from device code). Three things conspired:

1. CMake generates a device-link step (`cmake_device_link.o`) that references
   `fatbinData@Base` as a dynamic relocation — it must resolve from a
   global `fatbinData` symbol inside the same `.so`.
2. The final link used `g++`, not `nvcc`, so `libcudart`/`libcudadevrt`
   weren't pulled in automatically.
3. `pybind11_add_module` silently added `-flto=auto` (thin LTO). LTO
   reduced the local `fatbinData` in `spline_psf_gpu.cu.o` to file-scope
   and stripped it from the `.so`'s dynamic symbol table, making the
   relocation unresolvable at `dlopen()`.

**Fix:** Three changes to the pybind target:

```cmake
pybind11_add_module(spline NO_EXTRAS src/pybind_spline.cpp)

set_target_properties(spline PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
```

- `NO_EXTRAS` disables pybind11's automatic LTO (not overridable via
  `INTERPROCEDURAL_OPTIMIZATION` because pybind11 applies LTO through its
  own interface library, `pybind11::thin_lto`).
- `LINKER_LANGUAGE CUDA` makes nvcc the final linker, so it handles device
  linking and pulls in `libcudart`/`libcudadevrt` automatically.
- `CUDA_RESOLVE_DEVICE_SYMBOLS ON` forces the device-link step on this
  target even though it has no direct `.cu` sources.

## History of Fixes (superseded)

The previous DECODE-PRISM snapshot forked directly from
[TuragaLab/SplinePSF](https://github.com/TuragaLab/SplinePSF) at commit
`58f10b5` and carried an additional patch:

- **`sm_37 → sm_89` override** for CUDA 12.x compatibility.

**Status: no longer needed.** The Haydnspass upstream already defaults to
`CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 90`, which covers Ada
(sm_89), Ampere (sm_86, sm_80), and beyond. If you need to restrict the
build to a single architecture, override at configure time:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
```

| GPU | Compute Capability |
|-----|--------------------|
| RTX 4000 Ada | 89 |
| RTX 3090 | 86 |
| A100 | 80 |
| V100 | 70 |

## Installation

```bash
# Must be on a GPU node with nvcc available.
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1

# From the PRISM repo root:
pixi run pip install ./repos/SplinePSF/python
```

## Verification

```bash
pixi run python -c "
import spline
print('CPU:', hasattr(spline, 'PSFWrapperCPU'))
print('CUDA:', spline.cuda_compiled)
"
# Expected: CPU: True | CUDA: True
```

## DECODE-Side Fix: CUDA Tensor Device Mismatch

**File:** `decode/simulation/psf_kernel.py` (`CubicSplinePSF`, in the
separate DECODE submodule at `repos/DECODE/`).

**Problem:** `CubicSplinePSF.forward()` crashes when called with CUDA tensors:
```
RuntimeError: Expected all tensors to be on the same device, but found at
least two devices, cuda:0 and cpu!
```

**Root cause:** The C++ pybind wrapper (`PSFWrapperCUDA.forward_frames()`)
accepts **numpy arrays**, not CUDA tensors, and copies data to GPU internally.
But `forward()`, `forward_rois()`, and `derivative()` do coordinate math that
mixes CPU-only internal tensors (`self.vx_size`, `self.ref0`) with
user-provided tensors that may be on CUDA.

**Fix:** In `forward()`, `forward_rois()`, and `derivative()`, move inputs to
CPU at entry, do coordinate math + C++ call on CPU, then `.to(_in_device)`
the output back to the original device.

**Note:** The `device='cuda:0'` constructor arg controls which GPU the C++
CUDA kernel runs on — it does not mean Python-level tensors should be on CUDA.
The C++ wrapper handles all CPU→GPU→CPU transfers internally.

This patch lives in the DECODE fork (`boazmohar/DECODE`), not here.

## Upstream

- **Repo:** https://github.com/Haydnspass/SplinePSF
- **Branch followed:** `master`
- **Original ancestor:** https://github.com/TuragaLab/SplinePSF
