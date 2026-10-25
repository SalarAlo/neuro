# neuro

`neuro` is a small C++23 machine learning library with matrix, layer, network, and training code. The included MNIST example learns to recognize handwritten digits.

## Build and run

You need CMake 3.21 or newer and a C++23 compiler with `<print>` support. From the repository root, run:

```sh
cmake -S . -B build
cmake --build build
./build/neuro
```

The program trains on 1,000 images for five epochs by default, then asks for a test image index. It prints the image, its predicted digit, and the actual label. Enter `-1` to quit. Run from the repository root so the program can find the MNIST files in `data/`.

Use `--samples 60000` to train on the full set, or `--samples N` and `--epochs N` to choose another run length. `--no-prompt` exits after training. Run `./build/neuro --help` for the options. Debug builds use `-Og` on GCC and Clang.

## Neovim and clangd

The project's `.clangd` file points clangd to the CMake compilation database in `build/debug`. Configure that build once before opening the project in Neovim:

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
```

If Neovim was already open, run `:LspRestart` or reopen the C++ file so clangd reloads the build flags.

## Profile with KCachegrind

KCachegrind reads Callgrind files. Plain `valgrind ./build/neuro` runs Memcheck and does not create one. For a short, reproducible profile with debug symbols, run from the repository root:

```sh
cmake -S . -B build/profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/profile
valgrind --tool=callgrind --callgrind-out-file=build/profile/callgrind.out ./build/profile/neuro --samples 100 --epochs 1 --no-prompt
kcachegrind build/profile/callgrind.out
```

The profile build uses optimization and debug symbols. Increase `--samples` if the short run does not capture enough training work. Wait for Valgrind to finish before opening the output file.

Pass the generated file to KCachegrind; starting `kcachegrind` without a file can leave an empty window. Do not add `--instr-atstart=no` unless you also enable instrumentation during the run (for example with `callgrind_control -i on`). Otherwise the file contains zero events. `--no-prompt` lets this example finish immediately after training so Callgrind writes its final data.

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
