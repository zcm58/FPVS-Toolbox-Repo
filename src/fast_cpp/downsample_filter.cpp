#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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
    py::array_t<double> result({channels, samples});
    auto r = result.mutable_unchecked<2>();
    auto d = data.unchecked<2>();
    auto b = coeffs.unchecked<1>();
    for (size_t c = 0; c < channels; ++c) {
        for (size_t i = 0; i < samples; ++i) {
            double val = 0.0;
            for (size_t k = 0; k < ncoeffs; ++k) {
                if (i >= k) val += b(k) * d(c, i - k);
            }
            r(c, i) = val;
        }
    }
    return result;
}

PYBIND11_MODULE(downsample_filter, m) {
    m.def("downsample", &downsample, "Downsample 2D array by selecting every nth sample");
    m.def("apply_fir_filter", &apply_fir_filter, "Apply FIR filter using convolution");
}
