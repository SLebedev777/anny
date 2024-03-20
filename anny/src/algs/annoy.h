#pragma once

#include <exception>
#include <memory>
#include <limits>
#include <random>
#include <ctime>
#include <unordered_set>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"
#include "../core/hyperplane.h"
#include "../utils/utils_defs.h"


namespace anny
{

	template <typename T, typename Dist>
	class Annoy: public IKnnAlgorithm<T>
	{
	public:
		Annoy(size_t num_trees = 100, size_t leaf_size = 40, 
			unsigned int seed = anny::utils::UNDEFINED_SEED)
			: m_num_trees(num_trees)
			, m_leaf_size(leaf_size)
			, m_gen(std::mt19937(seed != anny::utils::UNDEFINED_SEED ? seed : time(0)))  // TODO: seeding from time(0) is VERY BAD for multi-threaded (parallel) calcs 
		{}

		~Annoy() override {}

		void fit(const std::vector<std::vector<T>>& data) override;
		IndexVector knn_query(const std::vector<T>& vec, size_t k) override;
		IndexVector radius_query(const std::vector<T>& vec, T radius) override;

	private:
		struct Node;
		using NodePtr = std::unique_ptr<Node>;

		struct Node
		{
			virtual ~Node() {}

			Hyperplane<T> border;
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
			Hyperplane<T> border;        // splitting hyperplane
			IndexVector left_indices;    // indices of data points that lie on the one side of the splitting hyperplane
			IndexVector right_indices;   // indices of data points that lie on the other side of the splitting hyperplane
		};

		friend class NodeVisitor;

		struct NodeVisitResult
		{
			T margin{ std::numeric_limits<T>::infinity() };
			bool need_take_wrong_side{ false };
		};

		class NodeVisitor
		{
		public:
			virtual NodeVisitResult visit(Annoy<T, Dist>::Node* node) = 0;
			virtual std::vector<std::pair<T, index_t>> get_result() const = 0;
			virtual size_t get_num_candidates() const = 0;
			virtual size_t get_num_max_candidates() const = 0;
		};

		class KnnQueryNodeVisitor : public NodeVisitor
		{
		public:
			KnnQueryNodeVisitor(Annoy<T, Dist>* context, VecView<T> vec, size_t k)
				: m_context(context)
				, m_vec(vec)
				, m_k(k)
			{}

			NodeVisitResult visit(Annoy<T, Dist>::Node* node) override
			{
				if (!node)
					return {};

				if (node->is_leaf())
				{
					const auto& indices = static_cast<Annoy<T, Dist>::LeafNode*>(node)->indices;
					m_candidates.insert(indices.begin(), indices.end());
					return {};
				}

				return { node->border.margin(m_vec), true };
			}

			std::vector<std::pair<T, index_t>> get_result() const override
			{
				std::vector<anny::index_t> indices(m_candidates.begin(), m_candidates.end()); // copying here
				return m_context->calc_distances(m_vec, indices);
			}

			size_t get_num_candidates() const override
			{
				return m_candidates.size();
			}

			size_t get_num_max_candidates() const override
			{
				return m_k;
			}

		private:
			std::unordered_set<anny::index_t> m_candidates;
			Annoy<T, Dist>* m_context;
			VecView<T> m_vec;
			size_t m_k;
		};


		class RadiusQueryNodeVisitor : public NodeVisitor
		{
		public:
			RadiusQueryNodeVisitor(Annoy<T, Dist>* context, VecView<T> vec, T radius)
				: m_context(context)
				, m_vec(vec)
				, m_radius(radius)
			{}

			NodeVisitResult visit(Annoy<T, Dist>::Node* node) override
			{
				if (!node)
					return {};

				if (node->is_leaf())
				{
					const auto& indices = static_cast<Annoy<T, Dist>::LeafNode*>(node)->indices;
					m_candidates.insert(indices.begin(), indices.end());
					return {};
				}

				T margin = node->border.margin(m_vec);

				return { margin, std::fabs(margin) <= m_radius };
			}

			std::vector<std::pair<T, index_t>> get_result() const override
			{
				std::vector<anny::index_t> indices(m_candidates.begin(), m_candidates.end()); // copying here
				auto result = m_context->calc_distances(m_vec, indices);
				auto it = std::upper_bound(result.begin(), result.end(), m_radius, [](T value, const auto& item)
					{
						return value < item.first;
					});
				result.erase(it, result.end());
				return result;
			}

			size_t get_num_candidates() const override
			{
				return m_candidates.size();
			}

			size_t get_num_max_candidates() const override
			{
				return m_context->m_data.num_rows();
			}

		private:
			std::unordered_set<anny::index_t> m_candidates;
			Annoy<T, Dist>* m_context;
			VecView<T> m_vec;
			T m_radius;
		};


		bool split(const IndexVector& indices, SplitResult& result);
		NodePtr build_annoy_tree(const IndexVector& indices);
		T calc_distance(VecView<T> vec, index_t index);
		std::vector<std::pair<T, index_t>> calc_distances(VecView<T> vec, const IndexVector& indices);
		void traverse(VecView<T> vec, NodeVisitor& visitor);

	private:
		Matrix<T, MatrixStorageVV<T>> m_data;
		std::vector<NodePtr> m_forest;
		size_t m_num_trees;
		size_t m_leaf_size;
		std::mt19937 m_gen;
		Dist m_dist_func;
	};


	template <typename T, typename Dist>
	bool Annoy<T, Dist>::split(const IndexVector& indices, typename Annoy<T, Dist>::SplitResult& res)
	{
		if (indices.size() < 2)
			return false;

		if (indices.size() == 2 && (m_data[indices[0]] == m_data[indices[1]]))
		{
			return false;
		}

		size_t i1{0}, i2{1}; // for case indices.size() == 2
		if (indices.size() > 2)
		{
			// randomly select 2 different points
			std::uniform_int_distribution<size_t> dis(0, indices.size() - 1);
			i1 = dis(m_gen);
			for (i2 = 0; i2 < indices.size(); i2++)
			{
				if (m_data[indices[i1]] != m_data[indices[i2]])
					break;
			}
			if (i2 == indices.size())  // all given data points are equal, can't split
				return false;
		}
		const auto& v1 = m_data[indices[i1]];
		const auto& v2 = m_data[indices[i2]];

		// calculate border hyperplane going perpendicularly through the middle of these 2 points
		Vec<T> normal = v1 - v2;
		normal = anny::l2_normalize(normal.view());
		Hyperplane<T> border;
		if constexpr (std::is_same_v<Dist, anny::CosineDistance>)
		{
			border = std::move(Hyperplane<T>{ normal }); // for cosine metric, all splitting hyperplanes go through zero
		}
		else
		{
			Vec<T> midpoint = 0.5 * (v1 + v2);
			border = std::move(Hyperplane<T>{ normal, midpoint });
		}

		res.border = std::move(border);
		for (const auto& i: indices)
		{
			if (res.border.side(m_data[i]))
				res.right_indices.push_back(i);
			else
				res.left_indices.push_back(i);
		}

		return (res.left_indices.size() > 0 && res.right_indices.size() > 0); // return false if couldn't split into non-empty parts
	}


	template <typename T, typename Dist>
	typename Annoy<T, Dist>::NodePtr Annoy<T, Dist>::build_annoy_tree(const IndexVector& indices)
	{
		SplitResult split_res;

		if (indices.size() <= m_leaf_size || !Annoy<T, Dist>::split(indices, split_res))
		{
			auto node = std::make_unique<Annoy<T, Dist>::LeafNode>();
			node->indices = indices;
			return node;
		}

		auto node = std::make_unique<Annoy<T, Dist>::Node>();
		node->border = split_res.border;  // TODO: move
		node->left = std::move(build_annoy_tree(split_res.left_indices));
		node->right = std::move(build_annoy_tree(split_res.right_indices));
		return node;
	}


	template <typename T, typename Dist>
	void Annoy<T, Dist>::fit(const std::vector<std::vector<T>>& data)
	{
		MatrixStorageVV<T> storage(data);
		Matrix<T, MatrixStorageVV<T>> m(storage);
		m_data = std::move(m);

		if constexpr (std::is_same_v<Dist, anny::CosineDistance>)
		{
			anny::l2_normalize_inplace(m_data);
		}

		IndexVector all_indices(m_data.num_rows());
		std::iota(all_indices.begin(), all_indices.end(), 0);

		// TODO: do in parallel
		for (size_t i = 0; i < m_num_trees; i++)
		{
			NodePtr tree = build_annoy_tree(all_indices);
			m_forest.push_back(std::move(tree));
		}
		
	}

	template <typename T, typename Dist>
	T Annoy<T, Dist>::calc_distance(VecView<T> vec, index_t index)
	{
		return this->m_dist_func(m_data[index], vec);
	}


	template <typename T, typename Dist>
	std::vector<std::pair<T, index_t>> Annoy<T, Dist>::calc_distances(VecView<T> vec, const IndexVector& indices)
	{
		assert(m_data[0].is_same_size(vec));

		std::vector<std::pair<T, index_t>> distances;

		for (const auto& index : indices)
		{
			distances.push_back({ this->m_dist_func(m_data[index], vec), index});
		}

		std::stable_sort(distances.begin(), distances.end());

		return distances;
	}


	template <typename T, typename Dist>
	void Annoy<T, Dist>::traverse(VecView<T> vec, NodeVisitor& visitor)
	{
		// MaxHeap will sort nodes by margin in that way that we will always take node with the biggest positive margin
		std::priority_queue<std::pair<T, Annoy<T, Dist>::Node*>> pq;
		for (auto& root : m_forest)
			pq.push({ root->border.margin(vec), root.get() });
		
		const auto k = visitor.get_num_max_candidates(); // max total candidates for all trees in forest

		while (visitor.get_num_candidates() < k && !pq.empty())
		{
			auto [prev_margin, node] = pq.top();
			pq.pop();

			auto node_res = visitor.visit(node);
			if (!node->is_leaf())
			{
				T margin = node_res.margin;
				Annoy<T, Dist>::Node* good_side = node->right.get();
				Annoy<T, Dist>::Node* wrong_side = node->left.get();
				if (margin < 0)
				{
					margin = -margin;
					std::swap(good_side, wrong_side);
				}
				pq.push({ margin, good_side });
				if (node_res.need_take_wrong_side)
				{
					pq.push({ -margin, wrong_side });
				}
			}
		}
	}


	template <typename T, typename Dist>
	IndexVector Annoy<T, Dist>::knn_query(const std::vector<T>& vec, size_t k)
	{
		IndexVector result;
		if (k == 0)
			return result;

		const auto N = m_data.num_rows();
		k = (k > N) ? N : k;

		Vec<T> query(vec);
		
		if constexpr (std::is_same_v<Dist, anny::CosineDistance>)
		{
			anny::l2_normalize_inplace(query.view());
		}

		size_t num_candidates = k * m_num_trees;
		KnnQueryNodeVisitor visitor(this, query.view(), num_candidates);

		traverse(query.view(), visitor);
		auto candidates_vec = visitor.get_result();
		size_t actual_k = std::min(k, candidates_vec.size());
		std::transform(candidates_vec.begin(), candidates_vec.begin() + actual_k, std::back_inserter(result), [](auto el) { return el.second; });

		return result;

	}


	template <typename T, typename Dist>
	IndexVector Annoy<T, Dist>::radius_query(const std::vector<T>& vec, T radius)
	{
		IndexVector result;

		Vec<T> query(vec);
		if constexpr (std::is_same_v<Dist, anny::CosineDistance>)
		{
			anny::l2_normalize_inplace(query.view());
		}

		RadiusQueryNodeVisitor visitor(this, query.view(), radius);

		traverse(query.view(), visitor);
		auto candidates_vec = visitor.get_result();
		std::transform(candidates_vec.begin(), candidates_vec.end(), std::back_inserter(result), [](auto el) { return el.second; });

		return result;

	}

}