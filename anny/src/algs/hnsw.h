#pragma once

#include <exception>
#include <memory>
#include <limits>
#include <random>
#include <unordered_set>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"
#include "../core/graph.h"
#include "../utils/utils_defs.h"


namespace anny
{

	template <typename T, typename Dist = L2Distance>
	class HNSW: public IKnnAlgorithm<T>
	{
	public:
		HNSW(size_t M=16, size_t ef_construction=100, size_t ef_search=100)
			: m_gen{ std::mt19937(777) }
			, m_M{M}                             // number of element's neighbors at construction time
			, m_Mmax0{ 2 * M }                   // max number of element's neighbors at level 0
			, m_efConstruction{ef_construction}  // ef stands for "expansion factor"
			, m_efSearch{ef_search}
			, m_mL { 1.0 / log(1.0 * M) }        // norm factor to calc random level for an element
		{}

		~HNSW() override {}

		void fit(const std::vector<std::vector<T>>& data) override;
		IndexVector knn_query(const std::vector<T>& vec, size_t k) override;
		IndexVector radius_query(const std::vector<T>& vec, T radius) override;

		void set_ef_search(size_t ef) { m_efSearch = ef; }
		size_t get_ef_search() const noexcept { return m_efSearch; }

	private:
		// aliases
		using DI = anny::utils::DistIndexPair<T, index_t>;
		using PQ = anny::utils::UniqueFixedSizePriorityQueue<anny::utils::DistIndexPair<T, Dist>>;
		using level_t = int;

		// constants
		static constexpr level_t MAX_LAYERS = 4;  // TODO: take value from FAISS code

		// functions
		std::vector<DI> search_layer(VecView<T> q, const IndexVector& ep, size_t ef, size_t lc);
		void insert(index_t index);
		IndexVector select_neighbors(std::vector<DI> neighbors, size_t M, bool is_sorted=true) const noexcept;
		void shrink_connections(index_t index, size_t lc, size_t M);
		void clear();
		level_t get_random_level();
		bool is_hnsw_empty() const noexcept;
		IndexVector knn_search(VecView<T> q, size_t k);

		T calc_distance(VecView<T> vec, index_t index);
		std::vector<DI> calc_distances(VecView<T> vec, const IndexVector& indices);

	private:
		Matrix<T, MatrixStorageVV<T>> m_data;
		std::vector<Graph<index_t>> m_layers;
		std::vector<level_t> m_elementLevels;
		std::mt19937 m_gen;
		Dist m_dist_func;
		size_t m_M{ 0 };
		size_t m_Mmax0{ 0 };
		size_t m_efConstruction{ 0 };
		size_t m_efSearch{ 0 };
		double m_mL{ 0 };
		level_t m_maxLevel{ -1 };   // curr max level (top level) during construction
		index_t m_entryPoint{ 0 }; // curr entry point at top level during construction
	};


	template <typename T, typename Dist>
	void HNSW<T, Dist>::clear()
	{
		m_layers.clear();
		m_elementLevels.clear();
	}


	template <typename T, typename Dist>
	std::vector<typename HNSW<T, Dist>::DI> HNSW<T, Dist>::search_layer(VecView<T> q, const IndexVector& ep, size_t ef, size_t lc)
	{
		std::unordered_set<index_t> visited;

		auto PQGreater = [](const DI& left, const DI& right) {
			return left.first > right.first;
		};
		std::priority_queue<DI, std::vector<DI>, decltype(PQGreater)> candidates{ PQGreater };  // min heap
		anny::utils::FixedSizePriorityQueue<DI> w{ ef };  // found nearest neighbors

		for (const auto& index : ep)
		{
			visited.insert(index);
			auto dist_epq = calc_distance(q, index);
			candidates.push({ dist_epq, index });
			w.push({ dist_epq, index });
		}

		while (!candidates.empty())
		{
			auto [dist_cq, c] = candidates.top();
			candidates.pop();
			auto dist_fq = w.top().first;

			// All candidates are worse than collected nearest neighbors by now. Stop.
			if (dist_cq > dist_fq)
				break;

			for (const auto& e : m_layers[lc].get_adj_vertices(c))
			{
				if (auto it = visited.find(e); it != visited.end())
					continue;
				visited.insert(e);

				auto dist_eq = calc_distance(q, e);
				auto dist_fq = w.top().first;
				if (dist_eq < dist_fq || w.size() < ef)
				{
					candidates.push({ dist_eq, e });
					w.push({ dist_eq, e });
				}
			}
		}

		return anny::utils::pq2vec(std::move(w));
	}


	template <typename T, typename Dist>
	typename HNSW<T, Dist>::level_t HNSW<T, Dist>::get_random_level()
	{
		std::uniform_real_distribution<double> dis(0.0, 1.0);
		double r = -log(dis(m_gen)) * m_mL;
		level_t level = static_cast<level_t>(r);
		level = (level > MAX_LAYERS) ? MAX_LAYERS : level;
		return level;
	}


	template <typename T, typename Dist>
	bool HNSW<T, Dist>::is_hnsw_empty() const noexcept
	{
		if (m_layers.empty())
			return true;

		return m_layers.front().is_empty();  // layer 0 contains all existing vertices
	}


	template <typename T, typename Dist>
	IndexVector HNSW<T, Dist>::select_neighbors(std::vector<DI> neighbors, size_t M, bool is_sorted) const noexcept
	{
		if (!is_sorted)
		{
			std::stable_sort(neighbors.begin(), neighbors.end());  // maybe overkill, because almost always input is already sorted by distance
		}
		IndexVector result;
		M = std::min(M, neighbors.size());
		std::transform(neighbors.begin(), neighbors.begin() + M, std::back_inserter(result), [](const auto& item) {
			return item.second;
			});
		return result;
	}


	template <typename T, typename Dist>
	void HNSW<T, Dist>::shrink_connections(index_t index, size_t lc, size_t M)
	{
		const auto& neighbors_indices = m_layers[lc].get_adj_vertices(index);
		VecView<T> vec = m_data[index];
		auto neighbors_with_distances = calc_distances(vec, neighbors_indices);
		IndexVector selected_neighbors = select_neighbors(neighbors_with_distances, M, /*is_sorted*/ true);
		for (const auto& n : neighbors_indices)
		{
			if (auto it = std::find(selected_neighbors.begin(), selected_neighbors.end(), n); it == selected_neighbors.end())
				m_layers[lc].delete_edge(index, n);
		}
	}


	template <typename T, typename Dist>
	void HNSW<T, Dist>::insert(index_t index)
	{
		level_t insert_level = get_random_level();

		m_elementLevels[index] = insert_level;

		for (level_t lc = insert_level; lc >= 0; lc--)
		{
			auto& g = m_layers[lc];
			g.insert_vertex(index);
		}

		// first insertion on empty hnsw
		if (m_maxLevel == -1)
		{
			m_maxLevel = insert_level;
			m_entryPoint = index;
			return;
		}

		// insert here
		VecView<T> q = m_data[index];

		IndexVector ep = { m_entryPoint };
		level_t lc = m_maxLevel;
		// greedy search for finding nearest entry point at curr max level 
		for (; lc > insert_level; lc--)
		{
			auto search_res = search_layer(q, ep, /*ef*/ 1, lc);
			ep = { search_res.front().second }; // because we take only 1 closest neighbor on each of these layers
		}
		// insert vertex and add edges to closest neighbors
		for (lc = std::min(insert_level, m_maxLevel); lc >= 0; lc--)
		{
			auto search_res = search_layer(q, ep, m_efConstruction, lc);
			IndexVector neighbors = select_neighbors(search_res, m_M);

			auto& g = m_layers[lc];
			for (const auto& n : neighbors)
			{
				g.insert_edge(index, n);
			}

			// cut edges
			size_t curr_M = lc > 0 ? m_M : m_Mmax0;
			for (const auto& n : neighbors)
			{
				if (g.get_adj_vertices(n).size() > curr_M)
				{
					shrink_connections(n, lc, curr_M);
				}
			}
		}
		
		if (insert_level > m_maxLevel)
		{
			m_maxLevel = insert_level;
			m_entryPoint = index;
		}
	}


	template <typename T, typename Dist>
	IndexVector HNSW<T, Dist>::knn_search(VecView<T> q, size_t k)
	{
		IndexVector ep = { m_entryPoint };
		level_t lc = m_maxLevel;
		// greedy search until level 1 
		for (; lc >= 1; lc--)
		{
			auto search_res = search_layer(q, ep, /*ef*/ 1, lc);
			ep = { search_res.front().second }; // because we take only 1 closest neighbor on each of these layers
		}
		// search at level 0
		auto search_res = search_layer(q, ep, m_efSearch, 0);
		IndexVector result = select_neighbors(search_res, k, /*is_sorted*/ true);
		return result;
	}


	template <typename T, typename Dist>
	void HNSW<T, Dist>::fit(const std::vector<std::vector<T>>& data)
	{
		MatrixStorageVV<T> storage(data);
		Matrix<T, MatrixStorageVV<T>> m(storage);
		m_data = std::move(m);
		
		IndexVector all_indices(m_data.num_rows());
		std::iota(all_indices.begin(), all_indices.end(), 0);
		
		// fit here
		clear();
			
		m_elementLevels.resize(m_data.num_rows());
		for (level_t l = 0; l < MAX_LAYERS; l++)
			m_layers.push_back(anny::Graph<index_t>());

		for (size_t index = 0; index < m_data.num_rows(); index++)
		{
			insert(index);
		}

	}


	template <typename T, typename Dist>
	T HNSW<T, Dist>::calc_distance(VecView<T> vec, index_t index)
	{
		return this->m_dist_func(m_data[index], vec);
	}


	template <typename T, typename Dist>
	std::vector<typename HNSW<T, Dist>::DI> HNSW<T, Dist>::calc_distances(VecView<T> vec, const IndexVector& indices)
	{
		assert(m_data[0].is_same_size(vec));

		std::vector<DI> distances;

		for (const auto& index: indices)
		{
			distances.push_back( { this->m_dist_func(m_data[index], vec), index } );
		}

		std::stable_sort(distances.begin(), distances.end());

		return distances;
	}


	template <typename T, typename Dist>
	IndexVector HNSW<T, Dist>::knn_query(const std::vector<T>& vec, size_t k)
	{
		IndexVector result;
		if (k == 0)
			return result;

		const auto N = m_data.num_rows();
		k = (k > N) ? N : k;

		Vec<T> query(vec);

		result = knn_search(query.view(), k);

		return result;
	}

	template <typename T, typename Dist>
	IndexVector HNSW<T, Dist>::radius_query(const std::vector<T>& vec, T radius)
	{
		throw std::runtime_error("Not implemented");
	}

}