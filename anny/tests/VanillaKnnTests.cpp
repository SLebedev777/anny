#include <iostream>
#include <gtest/gtest.h>
#include "algs/vanilla_knn.h"

using namespace anny;

TEST(VanillaKnnTests, VanillaKnnTest0)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	VanillaKnn<double> alg(anny::l2_distance<double>);
	alg.fit(data);
	
	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg.knn_query(query, 3);
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}

	{
		std::vector<double> query = data[3];
		auto result = alg.knn_query(query, 10);
		std::vector<index_t> expected{ 3, 0, 2, 1 };
		EXPECT_EQ(result, expected);
	}

	{
		std::vector<double> query = { 0.0, 0.0 };
		auto result = alg.knn_query(query, data.size());
		std::vector<index_t> expected{ 0, 1, 2, 3 };
		EXPECT_EQ(result, expected);
	}

}

TEST(VanillaKnnTests, VanillaKnnTest1)
{
	std::vector<std::vector<double>> data = {
		{0.0, 0.0, 2.0},
		{1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0}
	};

	VanillaKnn<double> alg(anny::l2_distance<double>);
	alg.fit(data);

	std::vector<double> query = { 0.0, 0.0, 1.3 };
	auto result = alg.knn_query(query, 2);
	std::vector<index_t> expected{ 2, 0 };
	EXPECT_EQ(result, expected);

}

TEST(VanillaKnnTests, VanillaKnnRadiusTest1)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	VanillaKnn<double> alg(anny::l2_distance<double>);
	alg.fit(data);

	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg.radius_query(query, 5.0);
		std::vector<index_t> expected{ 0 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { 0.5, 0.0 };
		auto result = alg.radius_query(query, sqrt(2.0));
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}

}


