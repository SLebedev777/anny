#pragma once

#include <exception>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"


namespace anny
{

	template <typename T>
	class VanillaKnn: public IKnnAlgorithm<T>
	{
	public:
		VanillaKnn(DistanceFunc<T> dist_func)
			: IKnnAlgorithm<T>(dist_func)
		{}

		virtual ~VanillaKnn() {}

		void fit(const std::vector<std::vector<T>>& data) override;
		std::vector<index_t> knn_query(const std::vector<T>& vec, size_t k) override;
		std::vector<index_t> radius_query(const std::vector<T>& vec, T radius) override;
	
	private:
		std::vector<std::pair<index_t, T>> calc_distances(VecView<T> vec);

	private:
		Matrix<T, MatrixStorageVV<T>> m_data;
	};


	template <typename T>
	void VanillaKnn<T>::fit(const std::vector<std::vector<T>>& data)
	{
		MatrixStorageVV<T> storage(data);
		Matrix<T, MatrixStorageVV<T>> m(storage);
		m_data = std::move(m);
	}

	template <typename T>
	std::vector<index_t> VanillaKnn<T>::knn_query(const std::vector<T>& vec, size_t k)
	{
		std::vector<index_t> result;
		if (k == 0)
			return result;

		const auto N = m_data.num_rows();
		k = (k > N) ? N : k;

		Vec<T> query(vec);

		auto distances = this->calc_distances(query.view());

		std::transform(distances.begin(), distances.begin() + k, std::back_inserter(result), [](auto el) { return el.first; });
		return result;
	}

	template <typename T>
	std::vector<index_t> VanillaKnn<T>::radius_query(const std::vector<T>& vec, T radius)
	{
		std::vector<index_t> result;
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

	template <typename T>
	std::vector<std::pair<index_t, T>> VanillaKnn<T>::calc_distances(VecView<T> vec)
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