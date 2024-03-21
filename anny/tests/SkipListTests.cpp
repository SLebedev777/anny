#include <iostream>
#include <gtest/gtest.h>
#include "algs/skiplist.h"

using namespace anny;

TEST(SkipListTests, SkipListTest0)
{
	std::vector<std::pair<int, char>> data = {
		{1, '1'},
		{4, '4'},
		{5, '5'},
		{2, '2'},
		{7, '7'},
		{6, '6'},
		{9, '9'}
	};
	anny::experimental::SkipList<int, char> skiplist(data);
	skiplist.print();
	{
		auto result = skiplist.find(7);
		EXPECT_EQ(*result, '7');
	}
	{
		auto result = skiplist.find(0);
		EXPECT_FALSE(result);
	}
	{
		auto result = skiplist.find(8);
		EXPECT_FALSE(result);
	}
	{
		auto result = skiplist.find(3);
		EXPECT_FALSE(result);
	}


}