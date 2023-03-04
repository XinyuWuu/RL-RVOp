#include "pybind11_common_header.hpp"
#include "RVOcalculator.hpp"
#include "observator.hpp"
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(RVOcalculator, m)
{
     m.doc() = "Calculate the RVOplus";
     py::class_<RVO::RVOcalculator>(m, "RVOcalculator")
         .def(py::init<double, double, contours_t, points_t>(), "init function",
              py::arg("dmax"), py::arg("robot_r"), py::arg("contours"), py::arg("target"))
         .def("inLine", &RVO::RVOcalculator::inLine,
              "check if in line",
              py::arg("line"), py::arg("x0"))
         .def("inArc", &RVO::RVOcalculator::inArc,
              "check if in arc",
              py::arg("arc"), py::arg("x0"))
         .def("LineP", &RVO::RVOcalculator::LineP,
              "return posible points in line",
              py::arg("line"), py::arg("xr"), py::arg("points"))
         .def("ArcP", &RVO::RVOcalculator::ArcP,
              "return posible points in arc",
              py::arg("arc"), py::arg("xr"), py::arg("points"))
         .def("RVOplus", static_cast<rvop_t (RVO::RVOcalculator::*)(lines_t lines, arcs_t arcs, point_t xr, point_t vr, point_t vo)>(&RVO::RVOcalculator::RVOplus),
              "return rvop",
              py::arg("lines"), py::arg("arcs"),
              py::arg("xr"), py::arg("vr"),
              py::arg("vo"))
         .def("get_obs", &RVO::RVOcalculator::get_obs,
              "get observation of all robot",
              py::arg("posvel"));
}

PYBIND11_MODULE(Observator, m)
{
     m.doc() = "get observation";
     py::class_<OBS::Observator>(m, "Observator")
         .def(py::init<double, double>(), "init the Observation with robot info",
              py::arg("dmax"), py::arg("robot_r"))
         .def("set_model", &OBS::Observator::set_model, "set contours and targets",
              py::arg("contours"), py::arg("target"))
         .def("set_reward", &OBS::Observator::set_reward, "set reward paramters",
              "robot_r"_a, "vmax"_a, "rmax"_a, "tolerance"_a, "a"_a, "b"_a, "c"_a, "d"_a, "e"_a, "f"_a, "g"_a, "eta"_a, "h"_a, "mu"_a)
         .def("change_robot", &OBS::Observator::change_robot, "change robot dmax and radius",
              py::arg("dmax"), py::arg("robot_r"))
         .def("get_obs", &OBS::Observator::get_obs, "get observations",
              py::arg("posvel"));
}
