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


TEST(AnnoyTests, AnnoyManyTreesTest)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	Annoy<double> alg1(anny::distance_func_factory<double>(anny::DistanceId::L2), 100, 1);
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


TEST(AnnoyTests, AnnoyTestIris1)
{
	using namespace anny::utils;
	CSVLoadingSettings settings(',');
	auto data = load_csv<double>("datasets/iris.data.csv", settings);

	Annoy<double> alg(anny::distance_func_factory<double>(anny::DistanceId::L2), 100, 1);
	alg.fit(data);

	{
		for (index_t query_index = 0; query_index < data.size(); query_index++)
		{
			std::vector<double> query = data[query_index];
			
			{
				auto result = alg.knn_query(query, 1);
				EXPECT_EQ(data[result.front()], data[query_index]); // Iris data has duplicates, so we should expect vector equality, not indices
			}
			
			{
				auto result = alg.knn_query(query, 10);  // top-N
				EXPECT_EQ(data[result.front()], data[query_index]);
				size_t count = std::count(result.begin(), result.end(), query_index);
				EXPECT_EQ(count, 1);
			}
			
		}
	}

	{
		index_t query_index = 0;
		std::vector<double> query = data[query_index];
		auto result = alg.knn_query(query, data.size());  // should find all data points
		EXPECT_EQ(result.size(), data.size());
	}

	Annoy<double> alg10(anny::distance_func_factory<double>(anny::DistanceId::L2), 100, 10);
	alg10.fit(data);

	{
		for (index_t query_index = 0; query_index < data.size(); query_index++)
		{
			std::vector<double> query = data[query_index];

			{
				auto result = alg10.knn_query(query, 1);
				EXPECT_EQ(data[result.front()], data[query_index]); // Iris data has duplicates, so we should expect vector equality, not indices
			}

			{
				auto result = alg10.knn_query(query, 10);  // top-N
				EXPECT_EQ(data[result.front()], data[query_index]);
				size_t count = std::count(result.begin(), result.end(), query_index);
				EXPECT_EQ(count, 1);
			}

		}
	}


}

