#include <pybind11/pybind11.h>

#include "core/cptoy.h"

namespace py = pybind11;

PYBIND11_MODULE(libcptoy, m)
{
    py::class_<CPToy> (m, "CPToy", "cptoy class")
        .def(py::init<int>(), py::return_value_policy::take_ownership)        
        .def("cuda_test", &CPToy::cuda_test)
        .def("mpi_test", &CPToy::mpi_test);
}
