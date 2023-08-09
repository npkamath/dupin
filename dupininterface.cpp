#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dupin.h"

namespace py = pybind11;

PYBIND11_MODULE(_dupin, m) {
    py::class_<dupinalgo>(m, "DupinAlgo")
        .def_readwrite("data", &dupinalgo::datum)
        .def_readwrite("cost_matrix", &dupinalgo::cost_matrix)
        .def_readwrite("num_bkps", &dupinalgo::num_bkps)
        .def_readwrite("num_timesteps", &dupinalgo::num_timesteps)
        .def_readwrite("num_parameters", &dupinalgo::num_parameters)
        .def(py::init<>()) // Default constructor
        .def("read_input", &dupinalgo::read_input)
        .def("initialize_cost_matrix", &dupinalgo::initialize_cost_matrix)
        .def("getTopDownBreakpoints", &dupinalgo::getTopDownBreakpoints); 
        


}
