#pragma once

#include <cmath>
#include "vec.h"
#include "vec_view.h"
#include "distance.h"


namespace anny
{
	/*
	* N-dimensional hyperplane is described by a N-dimensional normal vector with L2 norm = 1, and real-valued intercept.
	* Hyperplane equation is:
	*     dot(normal, v) + intercept = 0,
	* if N-dimensional vector v belongs to the plane.
	*/
	
	template <typename T>
	struct Hyperplane
	{
		Hyperplane() = default;

		explicit Hyperplane(const Vec<T>& _normal, T _intercept = T{ 0 })
			: normal{ _normal }
			, intercept{ _intercept }
		{
			assert(anny::is_l2_normalized(normal.view()));
		}

		Hyperplane(const Vec<T>& _normal, const Vec<T>& x0)
			: normal{ _normal }
			, intercept{ -anny::dot(_normal, x0) }
		{}
		
		T distance(VecView<T> v)
		{
			return std::fabs(anny::dot(normal.view(), v) + intercept);
		}
		
		T distance(VecView<const T> v) const
		{
			return std::fabs(anny::dot(normal.view(), v) + intercept);
		}
		
		bool side(VecView<T> v)
		{
			return (anny::dot(normal.view(), v) + intercept) >= 0.0;
		}

		bool side(VecView<const T> v) const
		{
			return (anny::dot(normal.view(), v) + intercept) >= 0.0;
		}

		size_t size() const
		{
			return normal.size();
		}

		Vec<T> normal;
		T intercept{};
	};


	template <typename T>
	anny::Hyperplane<T> make_orthogonal_hyperplane(size_t size, size_t normal_dim, T intercept = T{ 0.0 })
	{
		Vec<T> normal(size, T{ 0.0 });
		normal[normal_dim] = T{ 1.0 };
		return anny::Hyperplane<T>(normal, intercept);
	}

}