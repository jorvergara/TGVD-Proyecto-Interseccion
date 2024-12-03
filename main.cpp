#include <pybind11/pybind11.h>

namespace py = pybind11;

float some_fn(float x) {
    return x * 2;
}
class SomeClass {
    float multiplier;

public:
    SomeClass(float multiplier) : multiplier(multiplier) {}
    float multiply(float x) {
        return x * multiplier;
    }
};


PYBIND11_MODULE(module_name, handle) {
    handle.doc() = "Module docstring";
    handle.def("some_fn", &some_fn, "Function docstring");    

    py::class_<SomeClass>(handle, "PySomeClass").
        def(py::init<float>()).
        def("multiply", &SomeClass::multiply);

}