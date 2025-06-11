import numpy as np
from time import perf_counter
from scipy.signal import lfilter
from fast_cpp import apply_fir_filter, EXTENSION_AVAILABLE


def main():
    if not EXTENSION_AVAILABLE or apply_fir_filter is None:
        print("Extension not available")
        return

    channels = 4
    samples = 100000
    data = np.random.randn(channels, samples).astype(np.float64)
    coeffs = np.hamming(128).astype(np.float64)

    t0 = perf_counter()
    out_cpp = apply_fir_filter(data, coeffs)
    cpp_time = perf_counter() - t0

    t1 = perf_counter()
    out_py = np.stack([lfilter(coeffs, [1.0], ch)[:samples] for ch in data])
    py_time = perf_counter() - t1

    print(f"C++ time: {cpp_time:.6f}s")
    print(f"Python time: {py_time:.6f}s")
    if cpp_time < py_time:
        print("C++ implementation is faster")
    else:
        print("Python fallback is faster")


if __name__ == "__main__":
    main()
