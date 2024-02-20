#pragma once

#include <exception>
#include <memory>
#include <limits>
#include <random>
#include <ctime>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"
#include "../core/hyperplane.h"
#include "../utils/utils_defs.h"


namespace anny
{

	template <typename T>
	class Annoy: public IKnnAlgorithm<T>
	{
	public:
		Annoy(anny::DistanceFunc<T> dist_func, size_t num_trees = 100, size_t leaf_size = 40, 
			unsigned int seed = anny::utils::UNDEFINED_SEED)
			: IKnnAlgorithm<T>(dist_func)
			, m_num_trees(num_trees)
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

		class NodeVisitor
		{
		public:
			virtual void visit(Annoy<T>::Node* node) = 0;
			virtual T get_worst_distance() const = 0;
			virtual std::vector<std::pair<T, index_t>> get_result() const = 0;
		};

		class KnnQueryNodeVisitor : public NodeVisitor
		{
		public:
			using PQ = anny::utils::UniqueFixedSizePriorityQueue<std::pair<T, index_t>>;

			KnnQueryNodeVisitor(Annoy<T>* context, VecView<T> vec, size_t k)
				: m_candidates(anny::utils::FixedSizePriorityQueue<std::pair<T, index_t>>{ k })
				, m_context(context)
				, m_vec(vec)
				, m_k(k)
			{
			}

			void visit(Annoy<T>::Node* node) override
			{
				if (!node)
					return;

				if (node->is_leaf())
				{
					const auto& indices = static_cast<Annoy<T>::LeafNode*>(node)->indices;
					for (auto&& el : m_context->calc_distances(m_vec, indices))
					{
						m_candidates.push(el);
					}
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
			Annoy<T>* m_context;
			VecView<T> m_vec;
			size_t m_k;
		};

		SplitResult split(const IndexVector& indices);
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
	};


	template <typename T>
	typename Annoy<T>::SplitResult Annoy<T>::split(const IndexVector& indices)
	{
		if (indices.empty())
			return SplitResult{};

		// select 2 points
		std::uniform_int_distribution<size_t> dis(0, indices.size() - 1);
		size_t i1 = dis(m_gen);
		size_t i2{ i1 };
		while (i2 == i1)
		{
			i2 = dis(m_gen);
		}
		const auto& v1 = m_data[indices[i1]];
		const auto& v2 = m_data[indices[i2]];

		// calculate border hyperplane going perpendicularly through the middle of these 2 points
		Vec<T> normal = v1 - v2;
		Vec<T> midpoint = 0.5 * (v1 + v2);
		normal = l2_normalize(normal.view());
		Hyperplane<T> border{ normal, midpoint };

		SplitResult res;
		res.border = std::move(border);
		for (const auto& i: indices)
		{
			if (res.border.side(m_data[i]))
				res.right_indices.push_back(i);
			else
				res.left_indices.push_back(i);
		}

		return res;
	}


	template <typename T>
	typename Annoy<T>::NodePtr Annoy<T>::build_annoy_tree(const IndexVector& indices)
	{
		if (indices.size() <= m_leaf_size)
		{
			auto node = std::make_unique<Annoy<T>::LeafNode>();
			node->indices = indices;
			return node;
		}

		auto split_res = Annoy<T>::split(indices);
		auto node = std::make_unique<Annoy<T>::Node>();
		node->border = split_res.border;  // TODO: move
		node->left = std::move(build_annoy_tree(split_res.left_indices));
		node->right = std::move(build_annoy_tree(split_res.right_indices));
		return node;
	}


	template <typename T>
	void Annoy<T>::fit(const std::vector<std::vector<T>>& data)
	{
		MatrixStorageVV<T> storage(data);
		Matrix<T, MatrixStorageVV<T>> m(storage);
		m_data = std::move(m);

		IndexVector all_indices(m_data.num_rows());
		std::iota(all_indices.begin(), all_indices.end(), 0);

		// TODO: do in parallel
		for (size_t i = 0; i < m_num_trees; i++)
		{
			NodePtr tree = build_annoy_tree(all_indices);
			m_forest.push_back(std::move(tree));
		}
		
	}

	template <typename T>
	T Annoy<T>::calc_distance(VecView<T> vec, index_t index)
	{
		return this->m_dist_func(m_data[index], vec);
	}


	template <typename T>
	std::vector<std::pair<T, index_t>> Annoy<T>::calc_distances(VecView<T> vec, const IndexVector& indices)
	{
		assert(m_data[0].is_same_size(vec));

		std::vector<std::pair<T, index_t>> distances;

		for (const auto& index : indices)
		{
			distances.push_back({ this->m_dist_func(m_data[index], vec), index });
		}

		std::stable_sort(distances.begin(), distances.end());

		return distances;
	}


	template <typename T>
	void Annoy<T>::traverse(VecView<T> vec, NodeVisitor& visitor)
	{
		std::priority_queue<std::pair<T, Annoy<T>::Node*>> pq;
		for (auto& root : m_forest)
			pq.push({ root->border.margin(vec), root.get() });
		
		while (!pq.empty())
		{
			auto [margin, node] = pq.top();  // margin is signed distance from query point to current hyperplane
			pq.pop();

			if (node->is_leaf())
			{
				visitor.visit(node);
			}
			else
			{
				T distance_to_border = std::fabs(margin);

				Annoy<T>::Node* good_branch;
				Annoy<T>::Node* opposite_branch;
				if (margin < 0.0)
				{
					good_branch = node->left.get();
					opposite_branch = node->right.get();
				}
				else
				{
					good_branch = node->right.get();
					opposite_branch = node->left.get();
				}

				if (good_branch)
				{
					auto good_br_margin = !good_branch->is_leaf() ? good_branch->border.margin(vec) : std::numeric_limits<T>::infinity();
					pq.push({ good_br_margin, good_branch });
				}
				
				// shall we check the opposite branch for possible neighbors?
				if (opposite_branch && (distance_to_border < visitor.get_worst_distance()))
				{
					auto opp_br_margin = !opposite_branch->is_leaf() ? opposite_branch->border.margin(vec) : std::numeric_limits<T>::infinity();
					pq.push({ opp_br_margin, opposite_branch });
				}
				
			}

		}
	}


	template <typename T>
	IndexVector Annoy<T>::knn_query(const std::vector<T>& vec, size_t k)
	{
		IndexVector result;
		if (k == 0)
			return result;

		const auto N = m_data.num_rows();
		k = (k > N) ? N : k;

		Vec<T> query(vec);

		size_t num_candidates = k * m_num_trees;
		KnnQueryNodeVisitor visitor(this, query.view(), num_candidates);

		traverse(query.view(), visitor);
		auto candidates_vec = visitor.get_result();
		std::transform(candidates_vec.begin(), candidates_vec.begin() + k, std::back_inserter(result), [](auto el) { return el.second; });

		return result;

	}


	template <typename T>
	IndexVector Annoy<T>::radius_query(const std::vector<T>& vec, T radius)
	{
		throw std::runtime_error("Not implemented!");
	}

}