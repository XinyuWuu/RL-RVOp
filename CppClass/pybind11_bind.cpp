#include "pybind11_common_header.hpp"
#include "RVOcalculator.hpp"
#include "observator.hpp"
#include "ctrlConverter.hpp"
#include "simulator.hpp"
#include "environment.hpp"
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
              py::arg("posvel"), py::arg("avevel"));
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
              "robot_r"_a, "vmax"_a, "rmax"_a, "tolerance"_a, "a"_a, "b"_a, "c"_a, "d"_a, "e"_a, "f"_a, "g"_a, "eta"_a, "h"_a, "mu"_a, "remix"_a, "rm_middle"_a, "dmax"_a, "w"_a, "tb"_a)
         .def("change_robot", &OBS::Observator::change_robot, "change robot dmax and radius",
              py::arg("dmax"), py::arg("robot_r"))
         .def("get_obs", &OBS::Observator::get_obs, "get observations",
              py::arg("posvel"), py::arg("avevel"));
}

PYBIND11_MODULE(CtrlConverter, m)
{
     m.doc() = "converte the velocity control command to joint ctrl in mujoco";
     py::class_<CTRL::CtrlConverter>(m, "CtrlConverter")
         .def(py::init<double, double, double, double, double>(), "init function",
              "vmax"_a = 1, "tau"_a = 0.5, "wheel_r"_a = 0.04, "wheel_d"_a = 0.28, "gain"_a = 7)
         .def("v2ctrlbatchL", py::overload_cast<const posvels_t, const points_t>(&CTRL::CtrlConverter::v2ctrlbatchL), "converte a batch of vector v in local coordinate to ctrl in mujoco according to posvels",
              py::arg("posvels"), py::arg("vs"))
         .def("v2ctrlbatchG", py::overload_cast<const posvels_t, const points_t>(&CTRL::CtrlConverter::v2ctrlbatchG), "converte a batch of vector v in global coordinate to ctrl in mujoco according to posvels",
              py::arg("posvels"), py::arg("vs"))
         .def("get_rmax", &CTRL::CtrlConverter::get_rmax, "get max omega of robot rotation");
}

PYBIND11_MODULE(Simulator, m)
{
     m.doc() = "mujoco simulator";
     py::class_<SIM::Simulator>(m, "Simulator")
         .def(py::init<const char *, int, bool, int, int>(), "init function", "modelfile"_a, "Nrobot"_a,
              "isRender"_a, "W"_a, "H"_a)
         .def("step", py::overload_cast<std::vector<double>, int>(&SIM::Simulator::step), "step the environment",
              py::arg("ctrl"), py::arg("N"))
         .def("get_rgb", &SIM::Simulator::get_rgb, "get rendered rgb buffer memory view")
         .def("get_posvels", &SIM::Simulator::get_posvels, "get posvels buffer memory view")
         .def("render", &SIM::Simulator::render, "render once")
         .def("CloseGLFW", &SIM::Simulator::CloseGLFW, "clean GLFW windows and context");
}

PYBIND11_MODULE(Environment, m)
{
     m.doc() = "Environment interface";
     py::class_<ENV::Environment>(m, "Environment")
         .def(py::init<>(), "init function")
         .def(py::init<double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, bool, int, double, double>(), "init function",
              "robot_r"_a, "dmax"_a, "vmax"_a, "tau"_a, "wheel_r"_a, "wheel_d"_a, "gain"_a, "tolerance"_a, "a"_a, "b"_a, "c"_a, "d"_a, "e"_a, "f"_a, "g"_a, "eta"_a, "h"_a, "mu"_a, "rreach"_a, "dreach"_a, "remix"_a, "rm_middle"_a, "w"_a, "tb"_a)
         .def("setSim", &ENV::Environment::setSim, "init simulator",
              "modelfile"_a, "Nrobot"_a, "target"_a, "contour"_a, "isRender"_a, "W"_a, "H"_a)
         .def("setCtrl", &ENV::Environment::setCtrl, "init CtrlConverter",
              "vmax"_a = 1, "tau"_a = 0.5, "wheel_r"_a = 0.04, "wheel_d"_a = 0.28, "gain"_a = 7)
         .def("setRwd", &ENV::Environment::setRwd, "set reward paramters",
              "robot_r"_a, "vmax"_a, "rmax"_a, "tolerance"_a, "a"_a, "b"_a, "c"_a, "d"_a, "e"_a, "f"_a, "g"_a, "eta"_a, "h"_a, "mu"_a, "rreach"_a, "dreach"_a, "remix"_a, "rm_middle"_a, "dmax"_a, "w"_a, "tb"_a)
         .def("setRvop", &ENV::Environment::setRvop, "init RVOpcalculator",
              "dmax"_a, "robot_r"_a)
         .def("get_rgb", &ENV::Environment::get_rgb, "get rgb buffer")
         .def("get_posvels", &ENV::Environment::get_posvels, "get posvels buffer")
         .def("get_r", &ENV::Environment::get_r, "get r buffer")
         .def("get_rm", &ENV::Environment::get_rm, "get rm buffer")
         .def("get_d", &ENV::Environment::get_d, "get d buffer")
         .def("get_NNinput1", &ENV::Environment::get_NNinput1, "get NNinput1 buffer")
         .def("cal_NNinput1", &ENV::Environment::cal_NNinput1, "calculate NNinput1",
              "Nullfill"_a)
         .def("cal_obs", &ENV::Environment::cal_obs, "calculate observations",
              "avevel"_a)
         .def("cal_reward", &ENV::Environment::cal_reward, "calculate reward")
         .def("render", &ENV::Environment::render, "render once")
         .def("CloseGLFW", &ENV::Environment::CloseGLFW, "clean GLFW windows and context")
         .def("stepVL", &ENV::Environment::stepVL, "step with velocity command in local coordinate",
              "vs"_a, "N"_a, "n"_a)
         .def("stepVG", &ENV::Environment::stepVG, "step with velocity command in global coordinate",
              "vs"_a, "N"_a, "n"_a);
}
