#include <pybind11/pybind11.h>

#include "core/cptoy.h"

namespace py = pybind11;

PYBIND11_MODULE(libcptoy, m)
{
        py::class_<CPToy> (m, "CPToy", "cptoy class")
            .def(py::init<int>(),
                 py::return_value_policy::take_ownership,
             "size"_a, R"(
            Args:
                size:
                    array size                
        )")
        
        .def("cuda_test", &CPToy::cuda_test,
            "value"_a, R"(
            fill data on device
            
            Args:
                value: float to store
        )")
        .def("mpi_test", &CPToy::mpi_test,
            "value"_a, R"(
            reduce data
            
            Args:
                value: float to sum
        )");
}
