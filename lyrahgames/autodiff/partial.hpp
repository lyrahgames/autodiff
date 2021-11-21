#pragma once
#include <cmath>
#include <concepts>
//
#include <lyrahgames/xstd/forward.hpp>

namespace lyrahgames::autodiff {

namespace generic {

using namespace lyrahgames::xstd::generic;

template <typename T>
concept real = std::regular<T> && std::totally_ordered<T> &&
    requires(const T& x, const T& y, T& z) {
  { T(0) } -> identical<T>;
  { T(1) } -> identical<T>;
  { +x } -> identical<T>;
  { -x } -> identical<T>;
  { x + y } -> identical<T>;
  { x - y } -> identical<T>;
  { x* y } -> identical<T>;
  { x / y } -> identical<T>;
  { z += x } -> identical<T&>;
  { z -= x } -> identical<T&>;
  { z *= x } -> identical<T&>;
  { z /= x } -> identical<T&>;
};

}  // namespace generic

template <generic::real T>
struct partial {
  using real = T;

  constexpr partial() = default;
  // Allow implicit construction of constants for binary operations.
  constexpr partial(real x) : x{x} {}
  constexpr partial(real x, real dx) : x{x}, dx{dx} {}

  // Standard ordering for values by ignoring partials.

  // constexpr partial(const partial&) = default;
  // constexpr partial& operator=(const partial&) = default;

  friend constexpr auto operator==(const partial& x,
                                   const partial& y) noexcept {
    return x.x == y.x;
  }

  friend constexpr auto operator<=>(const partial& x,
                                    const partial& y) noexcept {
    return x.x <=> y.x;
  }

  real x{};
  real dx{};
};

template <generic::real real>
constexpr auto d(real x, real dx = 1) noexcept -> partial<real> {
  return {x, dx};
}

template <generic::real real>
constexpr auto operator+(const partial<real>& x) noexcept -> partial<real> {
  return x;
}

template <generic::real real>
constexpr auto operator-(const partial<real>& x) noexcept -> partial<real> {
  return {-x.x, -x.dx};
}

template <generic::real real>
constexpr auto operator+(const partial<real>& x,
                         const partial<real>& y) noexcept -> partial<real> {
  return {x.x + y.x, x.dx + y.dx};
}

template <generic::real real>
constexpr auto operator+(real x, const partial<real>& y) noexcept
    -> partial<real> {
  return {x + y.x, y.dx};
}

template <generic::real real>
constexpr auto operator+(const partial<real>& x, real y) noexcept
    -> partial<real> {
  return y + x;
}

template <generic::real real>
constexpr auto operator-(const partial<real>& x,
                         const partial<real>& y) noexcept -> partial<real> {
  return {x.x - y.x, x.dx - y.dx};
}

template <generic::real real>
constexpr auto operator-(real x, const partial<real>& y) noexcept
    -> partial<real> {
  return {x - y.x, -y.dx};
}

template <generic::real real>
constexpr auto operator-(const partial<real>& x, real y) noexcept
    -> partial<real> {
  return {x.x - y, x.dx};
}

template <generic::real real>
constexpr auto operator*(const partial<real>& x,
                         const partial<real>& y) noexcept -> partial<real> {
  return {x.x * y.x, x.dx * y.x + x.x * y.dx};
}

template <generic::real real>
constexpr auto operator*(real x, const partial<real>& y) noexcept
    -> partial<real> {
  return {x * y.x, x * y.dx};
}

template <generic::real real>
constexpr auto operator*(const partial<real>& x, real y) noexcept
    -> partial<real> {
  return y * x;
}

template <generic::real real>
constexpr auto operator/(const partial<real>& x,
                         const partial<real>& y) noexcept -> partial<real> {
  const auto t = 1 / y.x;
  return {x.x * t, (x.dx * y.x - x.x * y.dx) * t * t};
}

template <generic::real real>
constexpr auto operator/(real x, const partial<real>& y) noexcept
    -> partial<real> {
  const auto t = 1 / y.x;
  return {x * t, -x * y.dx * t * t};
}

template <generic::real real>
constexpr auto operator/(const partial<real>& x, real y) noexcept
    -> partial<real> {
  const auto t = 1 / y;
  return {x.x * t, x.dx * t};
}

template <generic::real real>
constexpr auto operator+=(partial<real>& x, const partial<real>& y) noexcept
    -> partial<real>& {
  x.x += y.x;
  x.dx += y.dx;
  return x;
}

template <generic::real real>
constexpr auto operator+=(partial<real>& x, real y) noexcept -> partial<real>& {
  x.x += y;
  return x;
}

template <generic::real real>
constexpr auto operator-=(partial<real>& x, const partial<real>& y) noexcept
    -> partial<real>& {
  x.x -= y.x;
  x.dx -= y.dx;
  return x;
}

template <generic::real real>
constexpr auto operator-=(partial<real>& x, real y) noexcept -> partial<real>& {
  x.x -= y;
  return x;
}

template <generic::real real>
constexpr auto operator*=(partial<real>& x, const partial<real>& y) noexcept
    -> partial<real>& {
  x.dx = x.dx * y.x + x.x * y.dx;
  x.x *= y.x;
  return x;
}

template <generic::real real>
constexpr auto operator*=(partial<real>& x, real y) noexcept -> partial<real>& {
  x.x *= y;
  x.dx *= y;
  return x;
}

template <generic::real real>
constexpr auto operator/=(partial<real>& x, const partial<real>& y) noexcept
    -> partial<real>& {
  const auto t = 1 / y.x;
  x.dx = (x.dx * y.x - x.x * y.dx) * t * t;
  x.x *= t;
  return x;
}

template <generic::real real>
constexpr auto operator/=(partial<real>& x, real y) noexcept -> partial<real>& {
  x.x /= y;
  x.dx /= y;
  return x;
}

template <generic::real real>
constexpr auto sq(real x) -> real {
  return x * x;
}
template <generic::real real>
constexpr auto sq(const partial<real>& x) noexcept -> partial<real> {
  return {sq(x.x), 2 * x.x * x.dx};
}

template <generic::real real>
constexpr auto cb(real x) -> real {
  return sq(x) * x;
}
template <generic::real real>
constexpr auto cb(const partial<real>& x) noexcept -> partial<real> {
  const auto t = sq(x.x);
  return {t * x.x, 3 * t * x.dx};
}

using std::abs;
template <generic::real real>
constexpr auto abs(const partial<real>& x) noexcept -> partial<real> {
  return {abs(x.x),
          x.dx * ((x.x < 0) ? real(-1) : ((x.x > 0) ? real(1) : real(0)))};
}

using std::sqrt;
template <generic::real real>
constexpr auto sqrt(const partial<real>& x) noexcept -> partial<real> {
  const auto t = sqrt(x.x);
  return {t, x.dx / (2 * t)};
}

using std::cbrt;
template <generic::real real>
constexpr auto cbrt(const partial<real>& x) noexcept -> partial<real> {
  const auto t = cbrt(x.x);
  return {t, x.dx / (3 * t * t)};
}

using std::pow;
template <generic::real real>
constexpr auto pow(const partial<real>& x, const partial<real>& y) noexcept
    -> partial<real> {
  const auto t = pow(x.x, y.x - 1);
  const auto u = t * x.x;
  return {u, y.x * t * x.dx + log(x.x) * u * y.dx};
}
template <generic::real real>
constexpr auto pow(real x, const partial<real>& y) noexcept -> partial<real> {
  const auto t = pow(x, y.x);
  return {t, log(x) * t * y.dx};
}
template <generic::real real>
constexpr auto pow(const partial<real>& x, real y) noexcept -> partial<real> {
  const auto t = pow(x.x, y - 1);
  return {t * x.x, y * t * x.dx};
}

using std::exp;
template <generic::real real>
constexpr auto exp(const partial<real>& x) noexcept -> partial<real> {
  const auto t = exp(x.x);
  return {t, t * x.dx};
}

using std::log;
template <generic::real real>
constexpr auto log(const partial<real>& x) noexcept -> partial<real> {
  const auto t = exp(x.x);
  return {log(x.x), x.dx / x.x};
}

using std::cos;
using std::sin;

template <generic::real real>
constexpr auto sin(const partial<real>& x) noexcept -> partial<real> {
  return {sin(x.x), cos(x.x) * x.dx};
}

template <generic::real real>
constexpr auto cos(const partial<real>& x) noexcept -> partial<real> {
  return {cos(x.x), -sin(x.x) * x.dx};
}

using std::tan;
template <generic::real real>
constexpr auto tan(const partial<real>& x) noexcept -> partial<real> {
  const auto t = tan(x.x);
  return {t, (1 + sq(t)) * x.dx};
}

using std::asin;
template <generic::real real>
constexpr auto asin(const partial<real>& x) noexcept -> partial<real> {
  return {asin(x.x), x.dx / sqrt(1 - sq(x.x))};
}

using std::acos;
template <generic::real real>
constexpr auto acos(const partial<real>& x) noexcept -> partial<real> {
  return {acos(x.x), -x.dx / sqrt(1 - sq(x.x))};
}

using std::atan;
template <generic::real real>
constexpr auto atan(const partial<real>& x) noexcept -> partial<real> {
  return {atan(x.x), x.dx / (1 + sq(x.x))};
}

using std::cosh;
using std::sinh;

template <generic::real real>
constexpr auto sinh(const partial<real>& x) noexcept -> partial<real> {
  return {sinh(x.x), cosh(x.x) * x.dx};
}

template <generic::real real>
constexpr auto cosh(const partial<real>& x) noexcept -> partial<real> {
  return {cosh(x.x), sinh(x.x) * x.dx};
}

using std::tanh;
template <generic::real real>
constexpr auto tanh(const partial<real>& x) noexcept -> partial<real> {
  const auto t = tanh(x.x);
  return {t, (1 - sq(t)) * x.dx};
}

using std::asinh;
template <generic::real real>
constexpr auto asinh(const partial<real>& x) noexcept -> partial<real> {
  return {asinh(x.x), x.dx / sqrt(1 + sq(x.x))};
}

using std::acosh;
template <generic::real real>
constexpr auto acosh(const partial<real>& x) noexcept -> partial<real> {
  return {acosh(x.x), x.dx / sqrt(sq(x.x) - 1)};
}

using std::atanh;
template <generic::real real>
constexpr auto atanh(const partial<real>& x) noexcept -> partial<real> {
  return {atanh(x.x), x.dx / (1 - sq(x.x))};
}

}  // namespace lyrahgames::autodiff
