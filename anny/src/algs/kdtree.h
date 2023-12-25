#pragma once

#include <exception>
#include <memory>
#include "knn_abc.h"
#include "../core/vec_view.h"
#include "../core/matrix.h"
#include "../core/distance.h"


namespace anny
{

	template <typename T>
	class KDTree: public IKnnAlgorithm<T>
	{
	public:
		KDTree(DistanceFunc<T> dist_func, size_t leaf_size=40)
			: IKnnAlgorithm<T>(dist_func)
			, m_leaf_size(leaf_size)
		{}

		virtual ~KDTree() {}

		void fit(const std::vector<std::vector<T>>& data) override;
		std::vector<index_t> knn_query(const std::vector<T>& vec, size_t k) override;
		std::vector<index_t> radius_query(const std::vector<T>& vec, T radius) override;
	
	private:
		struct Node;
		using NodePtr = std::unique_ptr<Node>;

		struct Node
		{
			virtual ~Node() {}

			T split;
			NodePtr left = nullptr;
			NodePtr right = nullptr;

			bool is_leaf() const { return left == nullptr && right == nullptr; }
		};

		struct LeafNode : Node
		{
			~LeafNode() override {}

			std::vector<anny::index_t> indices;
		};
		
		struct SplitResult
		{
			T split;
			std::vector<anny::index_t> left_indices;
			std::vector<anny::index_t> right_indices;
		};

		SplitResult split(const std::vector<anny::index_t>& indices, size_t dim);
		NodePtr build_kdtree(size_t dim, size_t leaf_size, const std::vector<anny::index_t>& indices);


	private:
		Matrix<T, MatrixStorageVV<T>> m_data;
		NodePtr m_tree;
		size_t m_leaf_size;
	};

	template <typename T>
	typename anny::KDTree<T>::SplitResult anny::KDTree<T>::split(const std::vector<anny::index_t>& indices, size_t dim)
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
		auto split_value = dim_values[mid].second;
		SplitResult res;
		res.split = split_value;
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
	typename anny::KDTree<T>::NodePtr anny::KDTree<T>::build_kdtree(size_t dim, size_t leaf_size, const std::vector<anny::index_t>& indices)
	{
		if (indices.size() <= leaf_size)
		{
			auto node = std::make_unique<anny::KDTree<T>::LeafNode>();
			node->split = 0;
			node->left = nullptr;
			node->right = nullptr;
			node->indices = indices;
			return node;
		}

		dim %= m_data.num_cols();
		anny::KDTree<T>::SplitResult split_res = split(indices, dim);
		auto node = std::make_unique<anny::KDTree<T>::Node>();
		node->split = split_res.split;
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
		
		std::vector<anny::index_t> all_indices(m_data.num_rows());
		std::iota(all_indices.begin(), all_indices.end(), 0);
		m_tree = build_kdtree(0, m_leaf_size, all_indices);
	}

	template <typename T>
	std::vector<index_t> KDTree<T>::knn_query(const std::vector<T>& vec, size_t k)
	{
		throw std::runtime_error("Not implemented");
	}

	template <typename T>
	std::vector<index_t> KDTree<T>::radius_query(const std::vector<T>& vec, T radius)
	{
		throw std::runtime_error("Not implemented");
	}

}