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

	KDTree<double> alg1(anny::l2_distance<double>, 1);
	alg1.fit(data);
	/*
	{
		std::vector<double> query = { 5.0, 0.0 };
		auto result = alg.knn_query(query, 3);
		std::vector<index_t> expected{ 0, 1, 3 };
		EXPECT_EQ(result, expected);
	}
	*/

	KDTree<double> alg3(anny::l2_distance<double>, 3);  // leaf_size = 3
	alg3.fit(data);

}

TEST(KDTreeTests, KDTreeTestIris)
{
	anny::CSVLoadingSettings settings(',');
	auto data = anny::load_csv<double>("datasets/iris.data.csv", settings);

	KDTree<double> alg(anny::l2_distance<double>, 15);
	alg.fit(data);
}

