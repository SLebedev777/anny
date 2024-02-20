#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>

namespace anny
{
namespace utils
{

enum class BadLinesPolicy
{
	BL_RAISE,
	BL_WARN,
	BL_SKIP
};

constexpr size_t UNLIMITED = std::numeric_limits<size_t>::max();

struct CSVLoadingSettings
{
	explicit CSVLoadingSettings(
		char delimiter = ',',
		bool has_header = false,
		BadLinesPolicy bad_lines_policy = BadLinesPolicy::BL_RAISE,
		size_t n_rows = UNLIMITED
	)
		: delimiter(delimiter)
		, has_header(has_header)
		, bad_lines_policy(bad_lines_policy)
		, n_rows(n_rows)
	{}
	
	char delimiter;
	bool has_header;
	BadLinesPolicy bad_lines_policy;
	size_t n_rows;
};

namespace detail
{
	template <typename Iter, typename OutIter, typename T, typename SliceFunc>
	void split(Iter first, Iter last, OutIter out, const T& sep, SliceFunc slice_func)
	{
		while (first != last)
		{
			auto slice_end = std::find(first, last, sep);
			*out++ = slice_func(first, slice_end);
			if (slice_end == last)
				return;
			first = std::next(slice_end);
		}
	}
}

template <typename T>
std::vector<std::vector<T>> load_csv(const std::string& filename, const CSVLoadingSettings& settings)
{
	std::ifstream file(filename);
	if (!file)
		throw std::runtime_error("Failed to open input CSV file: " + filename);

	std::string line;
	bool is_first_line = true;
	size_t i = 0;
	size_t n_cols = 0;

	std::vector<std::vector<T>> data;

	while (std::getline(file, line))
	{
		if (settings.has_header)
			continue;
		if (i > settings.n_rows)
			break;

		try
		{
			std::vector<T> row;
			anny::utils::detail::split(line.begin(), line.end(), std::back_inserter(row), settings.delimiter, 
				[](auto first, auto last) {
					std::string token(first, last);
					return static_cast<T>(std::stold(token));
				}
			);
			if (is_first_line)
			{
				is_first_line = false;
				n_cols = row.size();
			}
			if (row.size() != n_cols)
			{
				throw std::runtime_error("wrong number of columns in a row, should be " + std::to_string(n_cols));
			}
			data.push_back(row);
		}
		catch (std::exception& ex)
		{
			std::string message = "line " + std::to_string(i) + ": " + ex.what();
			switch (settings.bad_lines_policy)
			{
			case BadLinesPolicy::BL_RAISE:
				throw std::runtime_error("Error: " + message);
				break;
			case BadLinesPolicy::BL_WARN:
				std::cout << "Warning (line will be skipped): " + message << std::endl;
				break;
			case BadLinesPolicy::BL_SKIP:
			default: break;
			}
		}
		catch (...)
		{
			throw std::runtime_error("CSV reading error, line " + std::to_string(i) + ": unknown error");
		}
	}

	file.close();
	return data;
}

}
}