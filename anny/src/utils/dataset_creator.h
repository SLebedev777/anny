#pragma once

#include <random>
#include <vector>


namespace anny
{
namespace utils
{

template <typename T = double>
std::vector<std::vector<T>> make_uniform(size_t num_samples, size_t dim, T min_value, T max_value)
{
	std::vector<std::vector<T>> data;
	std::default_random_engine gen;
	std::uniform_real_distribution<T> uni_dis{ min_value, max_value };

	// uniformly sample data points within given bounding hypercube
	for (size_t i = 0; i < num_samples; i++)
	{
		std::vector<T> vec(dim);
		for (size_t d = 0; d < dim; d++)
		{
			vec[d] = uni_dis(gen);
		}
		data.push_back(std::move(vec));
	}
	return data;
}


template <typename T = double>
std::vector<std::vector<T>> make_clusters(size_t num_samples, size_t dim, size_t num_clusters, T cluster_std, T min_value, T max_value)
{
	std::vector<std::vector<T>> data;
	std::default_random_engine gen;
	std::normal_distribution<T> normal_dis{ T{0}, cluster_std };
	std::uniform_real_distribution<T> uni_dis{ min_value, max_value };

	std::vector<std::vector<T>> centers(num_clusters);

	// randomly choose centers within given bounding hypercube
	for (size_t c = 0; c < num_clusters; c++)
	{
		centers[c].reserve(dim);
		for (size_t d = 0; d < dim; d++)
		{
			centers[c].push_back(uni_dis(gen));
		}
	}

	// sample data points around centers according to given normal distribution
	while (data.size() < num_samples)
	{
		size_t curr_c = data.size() % num_clusters;
		std::vector<T> vec(dim);
		for (size_t d = 0; d < dim; d++)
		{
			T value{max_value + 1};
			while (value < min_value || value > max_value)
			{
				value = normal_dis(gen) + centers[curr_c][d];
			}
			vec[d] = value;
		}
		data.push_back(std::move(vec));
	}

	return data;
}

}
}