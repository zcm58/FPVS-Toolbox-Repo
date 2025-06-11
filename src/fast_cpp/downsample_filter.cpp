#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <valarray>

namespace py = pybind11;

using Complex = std::complex<double>;
using CArray = std::valarray<Complex>;

static inline void fft(CArray& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    CArray even = x[std::slice(0, N / 2, 2)];
    CArray  odd = x[std::slice(1, N / 2, 2)];
    fft(even);
    fft(odd);

    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k]       = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}

static inline void ifft(CArray& x) {
    x = x.apply(std::conj);
    fft(x);
    x = x.apply(std::conj) / static_cast<double>(x.size());
}

static inline size_t next_pow_two(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

py::array_t<double> downsample(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                               size_t factor) {
    auto buf = data.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
    size_t channels = buf.shape[0];
    size_t samples = buf.shape[1];
    size_t new_samples = samples / factor;
    py::array_t<double> result({channels, new_samples});
    auto r = result.mutable_unchecked<2>();
    auto a = data.unchecked<2>();
    for (size_t c = 0; c < channels; ++c) {
        for (size_t i = 0; i < new_samples; ++i) {
            r(c, i) = a(c, i * factor);
        }
    }
    return result;
}

py::array_t<double> apply_fir_filter(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                                     py::array_t<double, py::array::c_style | py::array::forcecast> coeffs) {
    auto dbuf = data.request();
    auto cbuf = coeffs.request();
    if (dbuf.ndim != 2 || cbuf.ndim != 1) throw std::runtime_error("Invalid array dimensions");
    size_t channels = dbuf.shape[0];
    size_t samples = dbuf.shape[1];
    size_t ncoeffs = cbuf.shape[0];

    size_t conv_len = samples + ncoeffs - 1;
    size_t fft_len = next_pow_two(conv_len);

    CArray h_fft(fft_len);
    for (size_t i = 0; i < ncoeffs; ++i)
        h_fft[i] = coeffs.unchecked<1>()[i];
    for (size_t i = ncoeffs; i < fft_len; ++i)
        h_fft[i] = 0.0;
    fft(h_fft);

    py::array_t<double> result({channels, samples});
    auto r = result.mutable_unchecked<2>();
    auto d = data.unchecked<2>();
    for (size_t c = 0; c < channels; ++c) {
        CArray x_fft(fft_len);
        for (size_t i = 0; i < samples; ++i)
            x_fft[i] = d(c, i);
        for (size_t i = samples; i < fft_len; ++i)
            x_fft[i] = 0.0;
        fft(x_fft);
        for (size_t i = 0; i < fft_len; ++i)
            x_fft[i] *= h_fft[i];
        ifft(x_fft);
        for (size_t i = 0; i < samples; ++i)
            r(c, i) = x_fft[i].real();
    }
    return result;
}

PYBIND11_MODULE(downsample_filter, m) {
    m.def("downsample", &downsample, "Downsample 2D array by selecting every nth sample");
    m.def("apply_fir_filter", &apply_fir_filter, "Apply FIR filter using FFT convolution");
}
