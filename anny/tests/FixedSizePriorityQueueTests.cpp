#include <iostream>
#include <gtest/gtest.h>
#include "utils/fixed_size_priority_queue.h"

using namespace anny;


TEST(FixedSizePriorityQueueTests, Test0)
{
    {
        FixedSizePriorityQueue<int> pq(4);
        for (const auto& i : { 1, 2, 3, 4, 5, 2 })
            pq.push(i);
        
        auto result = pq2vec(std::move(pq));
        
        std::vector<int> expected({ 1, 2, 2, 3 });
        EXPECT_EQ(result, expected);
    }

    {
        using item_type = std::pair<double, int>;  // {distance, point}
        FixedSizePriorityQueue<item_type> pq(5);   // will sort by pair.first
        std::initializer_list<item_type> items = { {3.0, 3}, {4.5, 4}, {1.5, 7}, {4.0, 8}, {3.9, 1}, {3.2, 6}, {4.5, 5} };
        for (const auto& item : items)
            pq.push(item);

        auto result = pq2vec(std::move(pq));

        std::vector<item_type> expected({ {1.5, 7}, {3.0, 3}, {3.2, 6}, {3.9, 1}, {4.0, 8} });
        EXPECT_EQ(result, expected);
    }
}