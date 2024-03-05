#include <iostream>
#include <gtest/gtest.h>
#include "algs/annoy.h"
#include "utils/csv_loader.h"
#include "core/distance.h"
#include "utils/dataset_creator.h"
#include <string>

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
	return; 

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

TEST(AnnoyTests, AnnoyTestRadiusQuery)
{
	std::vector<std::vector<double>> data = {
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
		{0.0, -1.0}
	};

	Annoy<double> alg1(anny::distance_func_factory<double>(anny::DistanceId::L2), 10, 1);
	alg1.fit(data);

	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg1.radius_query(query, 1.0);
		std::vector<index_t> expected{};
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg1.radius_query(query, 10.0);
		std::vector<index_t> expected{ 0, 1, 3, 2 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { -0.5, -1 };
		auto result = alg1.radius_query(query, 1.0);
		std::vector<index_t> expected{ 3 };
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { 0.5, 0 };
		auto result = alg1.radius_query(query, 1.4);
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}


	Annoy<double> alg3(anny::distance_func_factory<double>(anny::DistanceId::L2), 10, 3);  // leaf_size = 3
	alg3.fit(data);

	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg1.radius_query(query, 1.0);
		std::vector<index_t> expected{};
		EXPECT_EQ(result, expected);
	}
	{
		std::vector<double> query = { -0.5, -1 };
		auto result = alg1.radius_query(query, 1.0);
		std::vector<index_t> expected{ 3 };
		EXPECT_EQ(result, expected);
	}
}


TEST(AnnoyTests, AnnoyTestRandomDatasetClusters)
{
	return;

	std::cout << "Generating dataset..." << std::endl;
	auto data = anny::utils::make_clusters<double>(100000, 16, 1000, 1.0, -100.0, 100.0);

	std::cout << "Fitting..." << std::endl;
	Annoy<double> alg(anny::distance_func_factory<double>(anny::DistanceId::L2), 100, 1); // 100 trees, 1 point in leaf
	alg.fit(data);
	
	std::cout << "Searching..." << std::endl;
	std::default_random_engine gen;
	std::uniform_int_distribution<size_t> dis{ 0, data.size() - 1 };
	const size_t n = 1000;
	for (size_t i = 0; i < n; i++)
	{
		size_t query_index = dis(gen);
		std::vector<double> query = data[query_index];
		size_t top_n = query_index % 100 + 1;

		auto result = alg.knn_query(query, top_n);
		EXPECT_EQ(data[result.front()], data[query_index]);
		size_t count = std::count(result.begin(), result.end(), query_index);
		EXPECT_EQ(count, 1);
		EXPECT_EQ(result.size(), top_n);  // make sure we always take enough neighbors
	}
}

TEST(AnnoyTests, AnnoyTestRandomDatasetCosine)
{
	std::vector<anny::utils::GaussianCluster<double>> clusters = {
		{{-5, -5}, 1.0, 10},
		{{5, 5}, 1.0, 20}
	};
	auto data = anny::utils::make_clusters<double>(clusters, -100.0, 100.0);
	std::transform(data.begin(), data.end(), data.begin(), [](auto& vec) {
		return anny::l2_normalize(Vec<double>(vec).view()).data();
		}
	);

	Annoy<double> alg(anny::distance_func_factory<double>(anny::DistanceId::COSINE), 100, 1); // 100 trees, 1 point in leaf
	alg.fit(data);

	std::default_random_engine gen;
	size_t query_index_start = 0;
	for (size_t c = 0; c < clusters.size(); c++)
	{
		std::uniform_int_distribution<size_t> dis{ 0, clusters[c].num_points - 1 };
		size_t query_index = query_index_start + dis(gen);
		std::vector<double> query = data[query_index];
		size_t top_n = clusters[c].num_points;

		auto result = alg.knn_query(query, top_n);
		EXPECT_EQ(data[result.front()], data[query_index]);
		size_t count = std::count(result.begin(), result.end(), query_index);
		EXPECT_EQ(count, 1);
		EXPECT_EQ(result.size(), top_n);  // make sure we always take enough neighbors
		// make sure we find all points within cluster
		auto [min_it, max_it] = std::minmax_element(result.begin(), result.end());
		EXPECT_TRUE(*min_it >= query_index_start);
		EXPECT_TRUE(*max_it < query_index_start + clusters[c].num_points);

		query_index_start += clusters[c].num_points;
	}

}

TEST(AnnoyTests, AnnoyTestRandomDatasetUniform)
{
	auto data = anny::utils::make_uniform(1000, 2, -100.0, 100.0);
}
