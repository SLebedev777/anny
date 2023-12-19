#pragma once

#include <cmath>
#include <type_traits>
#include <functional>
#include "vec_view.h"

namespace anny
{
constexpr double PI = 3.14159265358979323846;  // pi is a part of standard only in C++ 20...

template<typename T>
static bool are_floats_equal(T f1, T f2) {
	return (std::fabs(f1 - f2) <= std::numeric_limits<T>::epsilon() * std::fmax(std::fabs(f1), std::fabs(f2)));
}

template <typename T>
T l2_norm_squared(const VecView<T>& vec)
{
	return vec.dot(vec);
}

template <typename T>
T l2_norm(const VecView<T>& vec)
{
	static_assert(!std::is_integral_v<T>, "It's impossible to calculate L2 norm for vector of integers!");
	return sqrt(vec.dot(vec));
}

template <typename T>
Vec<T> l2_normalize(const VecView<T>& vec)
{
	return vec / l2_norm(vec);
}

template <typename T>
T l2_distance_squared(const VecView<T>& v1, const VecView<T>& v2)
{
	auto sub = v1 - v2;
	return l2_norm_squared(sub.view());
}

template <typename T>
T l2_distance(const VecView<T>& v1, const VecView<T>& v2)
{
	auto sub = v1 - v2;
	return l2_norm(sub.view());
}

template <typename T>
T cosine_similarity(const VecView<T>& v1, const VecView<T>& v2, bool need_normalize=false)
{
	auto sim = dot(v1, v2);
	if (need_normalize)
	{
		sim /= l2_norm(v1) * l2_norm(v2);
	}
	return sim;
}

template <typename T>
T cosine_distance(const VecView<T>& v1, const VecView<T>& v2)
{
	constexpr T one{ 1 };
	return one - cosine_similarity(v1, v2, false);
}

enum class DistanceId: size_t
{
	L2 = 0,
	L2_SQUARED,
	COSINE,
	UNKNOWN = static_cast<size_t>(-1)
};

template <typename T>
using DistanceFunc = std::function<T(const VecView<T>& v1, const VecView<T>& v2)>;

template <typename T>
DistanceFunc<T> distance_func_factory(DistanceId dist_id)
{
	switch (dist_id)
	{
	case DistanceId::L2:
		return anny::l2_distance<T>;
	case DistanceId::COSINE:
		return anny::cosine_distance<T>;
	default:
		throw std::runtime_error("DistanceId unsupported");
	}
}

}