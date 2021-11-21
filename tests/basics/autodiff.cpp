#include <random>
//
#include <doctest/doctest.h>
//
#include <lyrahgames/xstd/math.hpp>
//
#include <lyrahgames/autodiff/autodiff.hpp>

using namespace std;
using namespace lyrahgames;

using doctest::Approx;
using xstd::pow;

static_assert(autodiff::generic::real<float>);
static_assert(autodiff::generic::real<double>);
static_assert(autodiff::generic::real<autodiff::partial<float>>);
static_assert(autodiff::generic::real<autodiff::partial<double>>);

const auto f = [](auto x, auto y) { return pow<2>(x) * y / sin(x) + cos(y); };

const auto df1 = [](auto x, auto y) {
  const auto t = sin(x);
  return (2 * x * y * t - pow<2>(x) * y * cos(x)) / pow<2>(t);
};

const auto df2 = [](auto x, auto y) { return pow<2>(x) / sin(x) - sin(y); };

SCENARIO("") {
  using real = float;
  mt19937 rng{random_device{}()};
  uniform_real_distribution<real> dist{-1, 1};
  const auto random = [&] { return dist(rng); };

  const size_t n = 1'000'000;
  for (size_t i = 0; i < n; ++i) {
    const auto x = random();
    const auto y = random();
    const auto dx = random();
    const auto dy = random();

    using autodiff::d;
    {
      const auto [value, deriv] = f(d(x, dx), d(y, dy));
      CHECK(Approx(value) == f(x, y));
      CHECK(Approx(deriv) == df1(x, y) * dx + df2(x, y) * dy);
    }
    {
      const auto [value, deriv] = f(d(x), y);
      CHECK(Approx(value) == f(x, y));
      CHECK(Approx(deriv) == df1(x, y));
    }
    {
      const auto [value, deriv] = f(x, d(y));
      CHECK(Approx(value) == f(x, y));
      CHECK(Approx(deriv) == df2(x, y));
    }
  }
}
