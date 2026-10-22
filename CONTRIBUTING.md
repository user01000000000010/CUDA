# Contributing to CUDA

Thank you for your interest in contributing to the **CUDA & Hopper sm_90 Acceleration Library**!

We welcome contributions covering high-performance CUDA kernels, bug fixes, algorithmic enhancements, and documentation.

---

## 🛠️ Development Workflow

### 1. Fork and Clone
1. Fork the repository on GitHub: `https://github.com/user01000000000010/CUDA`
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/CUDA.git
   cd CUDA
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/my-new-kernel
   ```

### 2. Building and Compilation
To compile the CUDA extensions locally:
```bash
pip install -e .
# or inplace build:
python setup.py build_ext --inplace
```

### 3. Testing and Verification
Before opening a pull request, run the PyTorch benchmark and numerical validation test suite:
```bash
python benchmark_pytorch.py
```
Ensure all verification checks pass with zero numerical divergence.

---

## 📐 Coding & Architecture Standards

To maintain state-of-the-art performance on modern NVIDIA architectures (Ampere `sm_80`, Ada `sm_89`, Hopper `sm_90`+):
- **Tile IR & Tensormodelling:** Avoid scalar thread-level address computation with modulo/division. Use `tile_ir::TensorView` and compile-time shape descriptors (`tile_ir.cuh`).
- **Asynchronous Data Transfers:** Utilize `cp.async` (16-byte aligned bulk copies) and double-buffering rather than blocking synchronous loads.
- **Hardware Barriers:** For Hopper (`sm_90`), favor fine-grained `mbarrier` over global `__syncthreads()`.
- **Numerical Stability:** Use FP32 register accumulators and online softmax algorithms for all attention mechanisms.

---

## 📬 Submitting a Pull Request (PR)

1. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat: add Blackwell sm_100 tensor core tile support"
   ```
2. Push to your fork:
   ```bash
   git push origin feature/my-new-kernel
   ```
3. Open a Pull Request against the `main` branch on GitHub.
4. Provide a brief summary of the changes, performance benchmark results, and GPU hardware tested on.
