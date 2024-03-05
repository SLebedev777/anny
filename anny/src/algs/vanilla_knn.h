#pragma once

#include <exception>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"


namespace anny
{

	template <typename T, typename Dist>
	class VanillaKnn: public IKnnAlgorithm<T>
	{
	public:
		VanillaKnn()
		{}

		~VanillaKnn() override {}

		void fit(const std::vector<std::vector<T>>& data) override;
		IndexVector knn_query(const std::vector<T>& vec, size_t k) override;
		IndexVector radius_query(const std::vector<T>& vec, T radius) override;
	
	private:
		std::vector<std::pair<index_t, T>> calc_distances(VecView<T> vec);

	private:
		Matrix<T, MatrixStorageVV<T>> m_data;
		Dist m_dist_func;
	};


	template <typename T, typename Dist>
	void VanillaKnn<T, Dist>::fit(const std::vector<std::vector<T>>& data)
	{
		MatrixStorageVV<T> storage(data);
		Matrix<T, MatrixStorageVV<T>> m(storage);
		m_data = std::move(m);
	}

	template <typename T, typename Dist>
	IndexVector VanillaKnn<T, Dist>::knn_query(const std::vector<T>& vec, size_t k)
	{
		IndexVector result;
		if (k == 0)
			return result;

		const auto N = m_data.num_rows();
		k = (k > N) ? N : k;

		Vec<T> query(vec);

		auto distances = this->calc_distances(query.view());

		std::transform(distances.begin(), distances.begin() + k, std::back_inserter(result), [](auto el) { return el.first; });
		return result;
	}

	template <typename T, typename Dist>
	IndexVector VanillaKnn<T, Dist>::radius_query(const std::vector<T>& vec, T radius)
	{
		IndexVector result;
		const auto N = m_data.num_rows();
		Vec<T> query(vec);

		auto distances = this->calc_distances(query.view());

		for (auto [i, dist] : distances)
		{
			if (dist > radius)
				break;
			result.push_back(i);
		}
		return result;
	}

	template <typename T, typename Dist>
	std::vector<std::pair<index_t, T>> VanillaKnn<T, Dist>::calc_distances(VecView<T> vec)
	{
		const auto N = m_data.num_rows();

		assert(m_data[0].is_same_size(vec));

		std::vector<std::pair<index_t, T>> distances(N);

		for (size_t i = 0; i < N; i++)
		{
			distances[i] = { i, this->m_dist_func(m_data[i], vec) };
		}

		std::stable_sort(distances.begin(), distances.end(), [](auto& left, auto& right) {
			return left.second < right.second;
			});

		return distances;
	}
}