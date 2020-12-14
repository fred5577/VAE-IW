#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "screen.h"
#include <vector>
#include <iostream>


// PYBIND11_MAKE_OPAQUE(std::vector<int>);

namespace py = pybind11;

std::vector<int>* calculate_bprost_features(std::vector<int> input1, std::vector<int> input2, bool bprost) {
    std::vector<int>* state = check_vector(input1, input2, bprost);
    return state;
}

double square(double x) {
    return x * x;
}

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(bprost, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    m.def("bprost", &calculate_bprost_features);

}