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

