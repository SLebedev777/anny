#include <iostream>
#include <gtest/gtest.h>
#include "utils/fixed_size_priority_queue.h"
#include "utils/unique_priority_queue.h"


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

TEST(UniquePriorityQueueTests, Test0)
{
    UniquePriorityQueue<int> upq(std::move(std::priority_queue<int>()));
    upq.push(1);
    upq.push(2);
    EXPECT_EQ(upq.size(), 2);
    upq.push(1);  // already exists, no insertion
    EXPECT_EQ(upq.size(), 2);
    upq.push(3);
    upq.pop();
    upq.pop();
    EXPECT_EQ(upq.size(), 1);
    EXPECT_EQ(upq.top(), 1);
    upq.push(2);  // now it's allowed to push 2
    EXPECT_EQ(upq.size(), 2);
    EXPECT_EQ(upq.top(), 2);
}

// TODO: test swap unique priority queue
TEST(UniquePriorityQueueTests, SwapTest)
{
    UniquePriorityQueue<int> upq(std::move(std::priority_queue<int>()));
    upq.push(1);
    upq.push(2);
    upq.push(3);
    UniquePriorityQueue<int> upq2(std::move(std::priority_queue<int>()));
    upq2.push(10);
    upq2.push(11);
    upq.swap(upq2);
    EXPECT_EQ(upq.size(), 2);
    EXPECT_EQ(upq.top(), 11);
    EXPECT_EQ(upq2.size(), 3);
    EXPECT_EQ(upq2.top(), 3);
}


namespace
{
    template <typename T>
    using FixedSizeUniquePriorityQueue = anny::UniquePriorityQueue<T, anny::FixedSizePriorityQueue<T>>;

}

TEST(FixedSizeUniquePriorityQueueTests, Test0)
{
    using item_type = std::pair<double, int>;
    FixedSizeUniquePriorityQueue<item_type> pq(std::move(anny::FixedSizePriorityQueue<item_type>{ 3 }));
    pq.push({ 3.0, 1 });
    pq.push({ 2.0, 2 });
    item_type item1{ 3.0, 1 };
    pq.push(item1);  // already exists, no insertion
    EXPECT_EQ(pq.size(), 2);
    EXPECT_EQ(pq.top(), item1);
    item_type item2{ 4.0, 3 };
    pq.push(item2);
    pq.push({ 5.0, 4 });  // not better than top, no insertion
    EXPECT_EQ(pq.size(), 3);
    EXPECT_EQ(pq.top(), item2);
}