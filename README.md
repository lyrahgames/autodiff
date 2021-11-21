<h1 align="center">
    Lyrahgames' Package for Automatic Differentiation
</h1>

<p align="center">
    C++ Header-Only Library Implementing a Little Bit of Automatic Differentiation
</p>

## Development Status

<p align="center">
    <img src="https://img.shields.io/github/languages/top/lyrahgames/autodiff.svg?style=for-the-badge">
    <img src="https://img.shields.io/github/languages/code-size/lyrahgames/autodiff.svg?style=for-the-badge">
    <img src="https://img.shields.io/github/repo-size/lyrahgames/autodiff.svg?style=for-the-badge">
    <a href="COPYING.md">
        <img src="https://img.shields.io/github/license/lyrahgames/autodiff.svg?style=for-the-badge&color=blue">
    </a>
</p>

<b>
<table align="center">
    <tr>
        <td>
            master
        </td>
        <td>
            <a href="https://github.com/lyrahgames/autodiff">
                <img src="https://img.shields.io/github/last-commit/lyrahgames/autodiff/master.svg?autodiffo=github&autodiffoColor=white">
            </a>
        </td>    
        <!-- <td>
            <a href="https://circleci.com/gh/lyrahgames/autodiff/tree/master"><img src="https://circleci.com/gh/lyrahgames/autodiff/tree/master.svg?style=svg"></a>
        </td> -->
        <!-- <td>
            <a href="https://codecov.io/gh/lyrahgames/autodiff">
              <img src="https://codecov.io/gh/lyrahgames/autodiff/branch/master/graph/badge.svg" />
            </a>
        </td> -->
        <td>
            <a href="https://ci.stage.build2.org/?builds=lyrahgames-autodiff&pv=&tc=*&cf=&mn=&tg=&rs=*">
                <img src="https://img.shields.io/badge/b|2 ci.stage.build2.org-Click here!-blue">
            </a>
        </td>
    </tr>
    <!-- <tr>
        <td>
            develop
        </td>
        <td>
            <a href="https://github.com/lyrahgames/autodiff/tree/develop">
                <img src="https://img.shields.io/github/last-commit/lyrahgames/autodiff/develop.svg?autodiffo=github&autodiffoColor=white">
            </a>
        </td>    
        <td>
            <a href="https://circleci.com/gh/lyrahgames/autodiff/tree/develop"><img src="https://circleci.com/gh/lyrahgames/autodiff/tree/develop.svg?style=svg"></a>
        </td>
        <td>
            <a href="https://codecov.io/gh/lyrahgames/autodiff">
              <img src="https://codecov.io/gh/lyrahgames/autodiff/branch/develop/graph/badge.svg" />
            </a>
        </td>
    </tr> -->
    <tr>
        <td>
        </td>
    </tr>
    <tr>
        <td>
            Current
        </td>
        <td>
            <a href="https://github.com/lyrahgames/autodiff">
                <img src="https://img.shields.io/github/commit-activity/y/lyrahgames/autodiff.svg?autodiffo=github&autodiffoColor=white">
            </a>
        </td>
        <!-- <td>
            <img src="https://img.shields.io/github/release/lyrahgames/autodiff.svg?autodiffo=github&autodiffoColor=white">
        </td>
        <td>
            <img src="https://img.shields.io/github/release-pre/lyrahgames/autodiff.svg?label=pre-release&autodiffo=github&autodiffoColor=white">
        </td> -->
        <td>
            <img src="https://img.shields.io/github/tag/lyrahgames/autodiff.svg?autodiffo=github&autodiffoColor=white">
        </td>
        <td>
            <img src="https://img.shields.io/github/tag-date/lyrahgames/autodiff.svg?label=latest%20tag&autodiffo=github&autodiffoColor=white">
        </td>
        <!-- <td>
            <a href="https://queue.cppget.org/autodiff">
                <img src="https://img.shields.io/website/https/queue.cppget.org/autodiff.svg?down_message=empty&down_color=blue&label=b|2%20queue.cppget.org&up_color=orange&up_message=running">
            </a>
        </td> -->
    </tr>
</table>
</b>

## Requirements
<b>
<table>
    <tr>
        <td>Language Standard:</td>
        <td>C++20</td>
    </tr>
    <tr>
        <td>Compiler:</td>
        <td>
            GCC | Clang
        </td>
    </tr>
    <tr>
        <td>Build System:</td>
        <td>
            <a href="https://build2.org/">build2</a>
        </td>
    </tr>
    <tr>
        <td>Operating System:</td>
        <td>
            Linux
        </td>
    </tr>
    <tr>
        <td>Dependencies:</td>
        <td>
            <a href="http://github.com/build2-packaging/doctest">
                doctest
            </a><br>
            <a href="http://github.com/lyrahgames/xstd">
                lyrahgames-xstd
            </a>
        </td>
    </tr>
</table>
</b>

## Getting Started

```c++
#include <random>
#include <cmath>
//
#include <lyrahgames/xstd/math.hpp>
//
#include <lyrahgames/autodiff/autodiff.hpp>

using namespace std;
using namespace lyrahgames;

using xstd::pow;
using autodiff::d;

// Provide a function template that can
// be computed for generic real types.
// Also ranges could be used instead.
constexpr auto f(auto x, auto y) noexcept {
  return pow<2>(x) * y / sin(x) + cos(y);
};

int main() {
  using real = float;

  // Create oracle for some random numbers.
  mt19937 rng{random_device{}()};
  uniform_real_distribution<real> dist{-1, 1};
  const auto random = [&] { return dist(rng); };

  // Get a random position and a random direction.
  const auto x = random();
  const auto y = random();
  const auto dx = random();
  const auto dy = random();

  {
    // Get value of f and its derivative in direction (dx,dy).
    // In this case, all the computations of f are done
    // with the template type 'partial<real>'.
    // This would also have to be done for general ranges.
    const auto [value, deriv] = f(d(x, dx), d(y, dy));
  }
  {
    // Get the value of f and its partial derivative in x direction.
    // Due to overloading, the compiler will generate a different
    // function evaluation and treat y as a constant.
    // Therefore already omitting unnecessary multiplications
    // with zero at compile time.
    const auto [value, deriv] = f(d(x), y);
  }
  {
    // Get the value of f and its partial derivative in y direction.
    const auto [value, deriv] = f(x, d(y));
  }
}

```

## Usage with build2
Add this repository to the `repositories.manifest` file of your build2 package.

    :
    role: prerequisite
    location: https://github.com/lyrahgames/autodiff.git

Add the following entry to the `manifest` file with a possible version dependency.

    depends: lyrahgames-autodiff

Add these entries to your `buildfile`.

    import libs = lyrahgames-autodiff%lib{lyrahgames-autodiff}
    exe{your-executable}: {hxx cxx}{**} $libs


## Installation
The standard installation process will only install the header-only library with some additional description, library, and package files.

    bpkg -d build2-packages cc \
      config.install.root=/usr/local \
      config.install.sudo=sudo

Get the latest package release and build it.

    bpkg build https://github.com/lyrahgames/autodiff.git

Install the built package.

    bpkg install lyrahgames-autodiff

For uninstalling, do the following.

    bpkg uninstall lyrahgames-autodiff

If your package uses an explicit `depends: lyrahgames-autodiff` make sure to initialize this dependency as a system dependency when creating a new configuration.

    bdep init -C @build cc config.cxx=g++ "config.cxx.coptions=-O3" -- "?sys:lyrahgames-autodiff/*"

## Alternative Usage
To use other build systems or manual compilation, you only have to add the `lyrahgames/autodiff/` directory to your project and include it in the compilation process.

