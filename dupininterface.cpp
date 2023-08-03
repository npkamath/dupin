/*#include <pybind11/pybind11.h>
#include "dupin.h"

namespace py = pybind11;

PYBIND11_MODULE(_dupin, m) {
    py::class_<dupinalgo>(m, "Dupin")
        .def(py::init<>())
        .def("read_input", &dupinalgo::read_input)
        .def("initialize_cost_matrix", &dupinalgo::initialize_cost_matrix, "initializes and fills cost matrix")
        .def("getTopDownBreakpoints", &dupinalgo::getTopDownBreakpoints, "calculates and stores breakpoints in a 2d vector");


}*/
