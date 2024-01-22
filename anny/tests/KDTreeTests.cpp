#include <iostream>
#include <gtest/gtest.h>
#include "algs/kdtree.h"
#include "utils/csv_loader.h"

using namespace anny;

TEST(KDTreeTests, KDTreeTest0)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	KDTree<double> alg1(1);
	alg1.fit(data);

	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg1.knn_query(query, 3);
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { -0.5, -1 };
		auto result = alg1.knn_query(query, 4);
		std::vector<index_t> expected{ 3, 2, 0, 1 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { 0.5, 0 };
		auto result = alg1.knn_query(query, 1);
		std::vector<index_t> expected{ 0 };
		EXPECT_EQ(result, expected);
	}


	KDTree<double> alg3(3);  // leaf_size = 3
	alg3.fit(data);

	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg1.knn_query(query, 3);
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { -0.5, -1 };
		auto result = alg1.knn_query(query, 4);
		std::vector<index_t> expected{ 3, 2, 0, 1 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { 0.5, 0 };
		auto result = alg1.knn_query(query, 1);
		std::vector<index_t> expected{ 0 };
		EXPECT_EQ(result, expected);
	}

}

TEST(KDTreeTests, KDTreeTestIris)
{
	anny::CSVLoadingSettings settings(',');
	auto data = anny::load_csv<double>("datasets/iris.data.csv", settings);

	KDTree<double> alg(15);
	alg.fit(data);

	{
		index_t query_index = 5;
		std::vector<double> query = data[query_index];
		auto result = alg.knn_query(query, 1);
		std::vector<index_t> expected{ query_index };
		EXPECT_EQ(result, expected);
	}

	{
		index_t query_index = 0;
		std::vector<double> query = data[query_index];
		auto result = alg.knn_query(query, data.size());  // should find all data points
		EXPECT_EQ(result.size(), data.size());
	}

}

