#pragma once

#include <vector>

namespace anny
{
	using index_t = size_t;  // index of an element in vector, or index of an object in dataset (row in data matrix)...
	using IndexVector = std::vector<index_t>;

	inline constexpr index_t UNDEFINED_INDEX = std::numeric_limits<size_t>::max();


	template <typename T>
	class IKnnAlgorithm
	{
	public:
		virtual ~IKnnAlgorithm() {}

		virtual void fit(const std::vector<std::vector<T>>& data) = 0;
		virtual IndexVector knn_query(const std::vector<T>& vec, size_t k) = 0;
		virtual IndexVector radius_query(const std::vector<T>& vec, T radius) = 0;	
	};

}