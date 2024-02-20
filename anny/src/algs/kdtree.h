#pragma once

#include <exception>
#include <memory>
#include <limits>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"
#include "../utils/utils_defs.h"


namespace anny
{

	template <typename T>
	class KDTree: public IKnnAlgorithm<T>
	{
	public:
		KDTree(size_t leaf_size=40)
			: IKnnAlgorithm<T>(anny::l2_distance<T>)  // only euclidean distance is supported
			, m_leaf_size(leaf_size)
		{}

		~KDTree() override {}

		void fit(const std::vector<std::vector<T>>& data) override;
		IndexVector knn_query(const std::vector<T>& vec, size_t k) override;
		IndexVector radius_query(const std::vector<T>& vec, T radius) override;
	
	private:
		struct Node;
		using NodePtr = std::unique_ptr<Node>;

		struct Node
		{
			virtual ~Node() {}

			T split{};
			anny::index_t split_index{ anny::UNDEFINED_INDEX };
			NodePtr left{ nullptr };
			NodePtr right{ nullptr };

			bool is_leaf() const { return left == nullptr && right == nullptr; }
		};

		struct LeafNode : Node
		{
			~LeafNode() override {}

			IndexVector indices;
		};
		
		struct SplitResult
		{
			T split;                     // median value along current splitting dimension
			IndexVector left_indices;    // indices of data points that lie to the left side of the splitting median value
			IndexVector right_indices;   // indices of data points that lie to the right side of the splitting median value
			anny::index_t split_index;   // index of splitting data point
		};

		friend class NodeVisitor;

		class NodeVisitor
		{
		public:
			virtual void visit(KDTree<T>::Node* node) = 0;
			virtual T get_worst_distance() const = 0;
			virtual std::vector<std::pair<T, index_t>> get_result() const = 0;
		};

		class KnnQueryNodeVisitor: public NodeVisitor
		{
		public:
			using PQ = anny::utils::UniqueFixedSizePriorityQueue<std::pair<T, index_t>>;

			KnnQueryNodeVisitor(KDTree<T>* tree, VecView<T> vec, size_t k)
				: m_candidates(anny::utils::FixedSizePriorityQueue<std::pair<T, index_t>>{ k })
				, m_tree(tree)
				, m_vec(vec)
				, m_k(k)
			{
			}

			void visit(KDTree<T>::Node* node) override
			{
				if (!node)
					return;

				if (node->is_leaf())
				{
					const auto& indices = static_cast<KDTree<T>::LeafNode*>(node)->indices;
					for (auto&& el : m_tree->calc_distances(m_vec, indices))
					{
						m_candidates.push(el);
					}
				}
				else
				{
					auto distance_to_split_point = m_tree->calc_distance(m_vec, node->split_index);
					m_candidates.push({ distance_to_split_point, node->split_index });
				}
			}

			T get_worst_distance() const override
			{
				return (!m_candidates.empty()) ? m_candidates.top().first : std::numeric_limits<T>::infinity();
			}

			std::vector<std::pair<T, index_t>> get_result() const override
			{
				PQ tmp(m_candidates);
				return anny::utils::pq2vec(std::move(tmp));
			}

		private:
			PQ m_candidates;
			KDTree<T>* m_tree;
			VecView<T> m_vec;
			size_t m_k;
		};


		class RadiusQueryNodeVisitor : public NodeVisitor
		{
		public:
			using PQ = anny::utils::UniquePriorityQueue<std::pair<T, index_t>>;

			RadiusQueryNodeVisitor(KDTree<T>* tree, VecView<T> vec, T radius)
				: m_candidates(std::priority_queue<std::pair<T, index_t>>{})
				, m_tree(tree)
				, m_vec(vec)
				, m_radius(radius)
			{
				assert(m_radius > 0.0);
			}

			void visit(KDTree<T>::Node* node) override
			{
				if (!node)
					return;

				if (node->is_leaf())
				{
					const auto& indices = static_cast<KDTree<T>::LeafNode*>(node)->indices;
					for (auto&& el : m_tree->calc_distances(m_vec, indices))
					{
						if (el.first <= m_radius)
							m_candidates.push(el);
					}
				}
				else
				{
					auto distance_to_split_point = m_tree->calc_distance(m_vec, node->split_index);
					if (distance_to_split_point <= m_radius)
						m_candidates.push({ distance_to_split_point, node->split_index });
				}
			}

			T get_worst_distance() const override
			{
				return m_radius;
			}

			std::vector<std::pair<T, index_t>> get_result() const override
			{
				PQ tmp(m_candidates);
				return anny::utils::pq2vec(std::move(tmp));
			}

		private:
			PQ m_candidates;
			KDTree<T>* m_tree;
			VecView<T> m_vec;
			T m_radius;
		};


		SplitResult split(const IndexVector& indices, size_t dim);
		NodePtr build_kdtree(size_t dim, size_t leaf_size, const IndexVector& indices);
		T calc_distance(VecView<T> vec, index_t index);
		std::vector<std::pair<T, index_t>> calc_distances(VecView<T> vec, const IndexVector& indices);
		void traverse_kdtree(KDTree<T>::Node* node, VecView<T> vec, size_t dim,	NodeVisitor& visitor);

	private:
		Matrix<T, MatrixStorageVV<T>> m_data;
		NodePtr m_tree;
		size_t m_leaf_size;
	};


	template <typename T>
	typename anny::KDTree<T>::SplitResult anny::KDTree<T>::split(const IndexVector& indices, size_t dim)
	{
		if (indices.empty())
			return SplitResult{};

		std::vector<std::pair<anny::index_t, T>> dim_values;  // pairs {row_index, data[row_index][dim]}
		std::transform(indices.begin(), indices.end(), std::back_inserter(dim_values),
			[this, dim](const auto& i) -> std::pair<anny::index_t, T> {
				return { i, m_data[i][dim] };
			});
		std::sort(dim_values.begin(), dim_values.end(), 
			[](const auto& left, const auto& right) {
				return left.second < right.second;
			});
		size_t mid = dim_values.size() / 2;
		auto split_value = dim_values[mid].second;  // split by median along dim
		SplitResult res;
		res.split = split_value;
		res.split_index = dim_values[mid].first;
		// TODO: optimize
		for (const auto& [i, value] : dim_values)
		{
			if (value < split_value)
				res.left_indices.push_back(i);
			else
				res.right_indices.push_back(i);
		}

		return res;
	}

	template <typename T>
	typename anny::KDTree<T>::NodePtr anny::KDTree<T>::build_kdtree(size_t dim, size_t leaf_size, const IndexVector& indices)
	{
		if (indices.size() <= leaf_size)
		{
			auto node = std::make_unique<anny::KDTree<T>::LeafNode>();
			node->indices = indices;
			return node;
		}

		dim %= m_data.num_cols();
		anny::KDTree<T>::SplitResult split_res = split(indices, dim);
		auto node = std::make_unique<anny::KDTree<T>::Node>();
		node->split = split_res.split;
		node->split_index = split_res.split_index;
		node->left = std::move(build_kdtree(dim + 1, leaf_size, split_res.left_indices));
		node->right = std::move(build_kdtree(dim + 1, leaf_size, split_res.right_indices));
		return node;
	}


	template <typename T>
	void KDTree<T>::fit(const std::vector<std::vector<T>>& data)
	{
		MatrixStorageVV<T> storage(data);
		Matrix<T, MatrixStorageVV<T>> m(storage);
		m_data = std::move(m);
		
		IndexVector all_indices(m_data.num_rows());
		std::iota(all_indices.begin(), all_indices.end(), 0);
		m_tree = build_kdtree(0, m_leaf_size, all_indices);
	}


	template <typename T>
	T KDTree<T>::calc_distance(VecView<T> vec, index_t index)
	{
		return this->m_dist_func(m_data[index], vec);
	}


	template <typename T>
	std::vector<std::pair<T, index_t>> KDTree<T>::calc_distances(VecView<T> vec, const IndexVector& indices)
	{
		assert(m_data[0].is_same_size(vec));

		std::vector<std::pair<T, index_t>> distances;

		for (const auto& index: indices)
		{
			distances.push_back( { this->m_dist_func(m_data[index], vec), index } );
		}

		std::stable_sort(distances.begin(), distances.end());

		return distances;
	}


	template <typename T>
	void KDTree<T>::traverse_kdtree(KDTree<T>::Node* node, VecView<T> vec, size_t dim, NodeVisitor& visitor)
	{
		if (!node)
			return;

		if (node->is_leaf())
		{
			visitor.visit(node);
			return;
		}
		else
		{
			dim %= m_data.num_cols();
			KDTree<T>::Node* good_branch;
			KDTree<T>::Node* opposite_branch;
			if (vec[dim] < node->split)
			{
				good_branch = node->left.get();
				opposite_branch = node->right.get();
			}
			else
			{
				good_branch = node->right.get();
				opposite_branch = node->left.get();
			}

			visitor.visit(node);

			traverse_kdtree(good_branch, vec, dim + 1, visitor);

			// shall we check the opposite branch for possible neighbors?
			auto distance_to_border = abs(vec[dim] - node->split);  // attention here - consistency with L2-distance metric is needed!!!
			auto worst_curr_distance = visitor.get_worst_distance();
			if (distance_to_border < worst_curr_distance)
			{
				traverse_kdtree(opposite_branch, vec, dim + 1, visitor);
			}
		}
	}


	template <typename T>
	IndexVector KDTree<T>::knn_query(const std::vector<T>& vec, size_t k)
	{
		IndexVector result;
		if (k == 0)
			return result;

		const auto N = m_data.num_rows();
		k = (k > N) ? N : k;

		Vec<T> query(vec);
		
		KnnQueryNodeVisitor visitor(this, query.view(), k);

		traverse_kdtree(m_tree.get(), query.view(), 0, visitor);
		auto candidates_vec = visitor.get_result();
		std::transform(candidates_vec.begin(), candidates_vec.end(), std::back_inserter(result), [](auto el) { return el.second; });

		return result;
	}

	template <typename T>
	IndexVector KDTree<T>::radius_query(const std::vector<T>& vec, T radius)
	{
		IndexVector result;

		Vec<T> query(vec);

		RadiusQueryNodeVisitor visitor(this, query.view(), radius);

		traverse_kdtree(m_tree.get(), query.view(), 0, visitor);
		auto candidates_vec = visitor.get_result();
		std::transform(candidates_vec.begin(), candidates_vec.end(), std::back_inserter(result), [](auto el) { return el.second; });

		return result;

	}

}