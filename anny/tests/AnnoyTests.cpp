#include <iostream>
#include <gtest/gtest.h>
#include "algs/annoy.h"
#include "utils/csv_loader.h"
#include "core/distance.h"

using namespace anny;

TEST(AnnoyTests, AnnoyBuildTreeTest)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	Annoy<double> alg1(anny::distance_func_factory<double>(anny::DistanceId::L2), 1, 1);
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

	
	Annoy<double> alg3(anny::distance_func_factory<double>(anny::DistanceId::L2), 1, 3);
	alg3.fit(data);

	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg3.knn_query(query, 3);
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { -0.5, -1 };
		auto result = alg3.knn_query(query, 4);
		std::vector<index_t> expected{ 3, 2, 0, 1 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { 0.5, 0 };
		auto result = alg3.knn_query(query, 1);
		std::vector<index_t> expected{ 0 };
		EXPECT_EQ(result, expected);
	}
	
}


