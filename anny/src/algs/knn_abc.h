#pragma once

#include <vector>
#include "../core/distance.h"

namespace anny
{
	using index_t = size_t;  // index of an element in vector, or index of an object in dataset (row in data matrix)...

	template <typename T>
	class IKnnAlgorithm
	{
	public:
		explicit IKnnAlgorithm(DistanceFunc<T> dist_func)
			: m_dist_func{ dist_func }
		{}

		virtual ~IKnnAlgorithm() {}

		virtual void fit(const std::vector<std::vector<T>>& data) = 0;
		virtual std::vector<index_t> knn_query(const std::vector<T>& vec, size_t k) = 0;
		virtual std::vector<index_t> radius_query(const std::vector<T>& vec, T radius) = 0;
	
		const DistanceFunc<T> m_dist_func;
	};

}