#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "dupin.h"

namespace py = pybind11;

PYBIND11_MODULE(_dupin, m) {
    py::class_<dupinalgo>(m, "DupinAlgo")
        .def_property("datum", &dupinalgo::getDatum, &dupinalgo::setDatum)
        .def_property("cost_matrix", &dupinalgo::getCostMatrix, &dupinalgo::setCostMatrix)
        .def_property("num_bkps", &dupinalgo::get_num_bkps, &dupinalgo::set_num_bkps)
        .def_property("num_timesteps", &dupinalgo::get_num_timesteps, &dupinalgo::set_num_timesteps)
        .def_property("num_parameters", &dupinalgo::get_num_parameters, &dupinalgo::set_num_parameters)
        .def(py::init<>()) // Default constructor
        .def("read_input", &dupinalgo::read_input)
        .def("initialize_cost_matrix", &dupinalgo::initialize_cost_matrix)
        .def("getTopDownBreakpoints", &dupinalgo::getTopDownBreakpoints);  
}
