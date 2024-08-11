#include <iostream>
#include <gtest/gtest.h>
#include "algs/hnsw.h"
#include "core/distance.h"
#include "utils/dataset_creator.h"
#include <string>

using namespace anny;

TEST(HNSWTests, HNSWBuildTestSimple)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	HNSW<double, L2Distance> alg1(/*M*/ 2, /*efConstruction*/ 2, /*efSearch*/ data.size());
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
	
}

TEST(HNSWTests, HNSWTestRandomDatasetClusters)
{
	//GTEST_SKIP();

	std::cout << "Generating dataset..." << std::endl;
	auto data = anny::utils::make_uniform<double>(10000, 16, -100.0, 100.0);

	std::cout << "Fitting..." << std::endl;
	HNSW<double, L2Distance> alg(/*M*/ 16, /*efConstruction*/ 100, /*efSearch*/ 100);
	alg.fit(data);

	std::cout << "Searching..." << std::endl;
	std::default_random_engine gen;
	std::uniform_int_distribution<size_t> dis{ 0, data.size() - 1 };
	const size_t n = 100;
	for (size_t i = 0; i < n; i++)
	{
		size_t query_index = dis(gen);
		std::vector<double> query = data[query_index];
		size_t top_n = 10;

		auto result = alg.knn_query(query, top_n);
		EXPECT_EQ(data[result.front()], data[query_index]);
		size_t count = std::count(result.begin(), result.end(), query_index);
		EXPECT_EQ(count, 1);
		EXPECT_EQ(result.size(), top_n);  // make sure we always take enough neighbors
	}
}
