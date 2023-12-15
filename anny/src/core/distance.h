#pragma once

#include <cmath>
#include <type_traits>
#include "vec_view.h"

namespace anny
{

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

}