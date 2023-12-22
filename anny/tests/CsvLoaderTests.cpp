#include <iostream>
#include <gtest/gtest.h>
#include "utils/csv_loader.h"


TEST(CsvLoaderTests, SplitTest)
{
	{
		std::string s{ "one,two,three" };
		const char sep{ ',' };
		std::vector<std::string> out;

		anny::detail::split(s.begin(), s.end(), std::back_inserter(out), sep, [](auto left, auto right) {
			return std::string(left, right);
			});
		EXPECT_EQ(out, std::vector<std::string>({ "one", "two", "three" }));
	}
	{
		std::string s{ "3.2;4.1;3.14;0.02;" };
		const char sep{ ';' };
		std::vector<double> out;

		anny::detail::split(s.begin(), s.end(), std::back_inserter(out), sep, [](auto left, auto right) {
			return std::stod(std::string(left, right));
			});
		EXPECT_EQ(out, std::vector<double>({ 3.2, 4.1, 3.14, 0.02 }));
		EXPECT_EQ(out.size(), 4);
	}

}

TEST(CsvLoaderTests, TestIris)
{
	anny::CSVLoadingSettings settings(',');
	auto data = anny::load_csv<float>("datasets/iris.data.csv", settings);
	const std::pair<size_t, size_t> shape{ data.size(), data.front().size() };
	const std::pair<size_t, size_t> shape_expected{ 150, 2 };
	EXPECT_EQ(shape, shape_expected);
}

TEST(CsvLoaderTests, TestDim128)
{
	anny::CSVLoadingSettings settings(',');
	auto data = anny::load_csv<char>("datasets/dim128.data.csv", settings);
	const std::pair<size_t, size_t> shape{ data.size(), data.front().size() };
	const std::pair<size_t, size_t> shape_expected{ 1024, 128 };
	EXPECT_EQ(shape, shape_expected);
}

