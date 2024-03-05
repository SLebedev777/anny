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


template <typename T = double>
struct GaussianCluster
{
	std::vector<T> center;
	T std;  // standard deviation
	size_t num_points;

	size_t dim() const { return center.size(); }
};


template <typename T = double>
std::vector<std::vector<T>> make_clusters(const std::vector<GaussianCluster<T>>& clusters, T min_value, T max_value)
{
	std::vector<std::vector<T>> data;
	std::default_random_engine gen;
	std::vector<std::normal_distribution<T>> normal_dis;

	const size_t dim = clusters.front().center.size();
	for (const auto& cl : clusters)
	{
		if (cl.dim() != dim)
			throw std::runtime_error("All clusters centers must have the same dimensionality");
	
		for (size_t d = 0; d < dim; d++)
		{
			if (cl.center[d] < min_value || cl.center[d] > max_value)
				throw std::runtime_error("All clusters centers must be within given min amd max values");
		}

		normal_dis.push_back(std::normal_distribution<T>(T{ 0 }, cl.std));
	}

	// sample data points around centers according to given normal distribution
	for (size_t c = 0; c < clusters.size(); c++)
	{
		for (size_t i = 0; i < clusters[c].num_points; i++)
		{
			std::vector<T> vec(dim);
			for (size_t d = 0; d < dim; d++)
			{
				T value{ max_value + 1 };
				while (value < min_value || value > max_value)
				{
					value = normal_dis[c](gen) + clusters[c].center[d];
				}
				vec[d] = value;
			}
			data.push_back(std::move(vec));
		}
	}

	return data;
}


}
}