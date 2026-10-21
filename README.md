# neuro

`neuro` is a small C++23 machine learning library with matrix, layer, network, and training code. The included MNIST example learns to recognize handwritten digits.

## Build and run

You need CMake 3.21 or newer and a C++23 compiler with `<print>` support. From the repository root, run:

```sh
cmake -S . -B build
cmake --build build
./build/neuro
```

The program trains for three epochs, then asks for a test image index. It prints the image, its predicted digit, and the actual label. Enter `-1` to quit. Run it from the repository root so it can find the MNIST files in `data/`.

## Use in another CMake project

Add this repository as a Git submodule in the other project's root:

```sh
git submodule add https://github.com/SalarAlo/ml_from_scratch.git vendor/neuro
```

Then add it to the other project's `CMakeLists.txt` and link your target:

```cmake
add_subdirectory(vendor/neuro)
target_link_libraries(your_app PRIVATE Neuro::core)
```

Include library headers through the `neuro/` prefix, for example `#include "neuro/Matrix.hpp"`, `#include "neuro/NeuralNetwork.hpp"`, or `#include "neuro/Activations.hpp"`. Public types live in the `Neuro` namespace; for example, `Neuro::Matrix<2, 2>` and `Neuro::Relu`. The MNIST example executable is built only when this repository is the top-level project.

To pull a newer version later, run `git submodule update --remote vendor/neuro`, then commit the updated submodule pointer in the other project. After cloning that project, run `git submodule update --init --recursive` to fetch its submodules.
