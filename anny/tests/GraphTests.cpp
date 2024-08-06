#include <iostream>
#include <gtest/gtest.h>
#include "core/graph.h"

using namespace anny;

TEST(GraphTests, GraphTestCreateFromMap)
{
	std::unordered_map<size_t, std::vector<size_t>> adj = {
        {0, {1}},
        {1, {0, 2, 3}},
        {2, {1, 3}},
        {3, {1, 2}}
	};

    Graph g(adj);

    EXPECT_EQ(g.num_vertices(), 4);
    EXPECT_EQ(g.num_edges(), 4);
    
    EXPECT_TRUE(g.has_vertex(2));
    EXPECT_FALSE(g.has_vertex(10));
    
    EXPECT_TRUE(g.has_edge(0, 1));
    EXPECT_TRUE(g.has_edge(1, 0));
    EXPECT_TRUE(g.has_edge(2, 3));
    EXPECT_TRUE(g.has_edge(3, 2));
    EXPECT_FALSE(g.has_edge(0, 3));
    EXPECT_FALSE(g.has_edge(2, 2));

    const std::vector<size_t> v = { 0, 2, 3 };
    EXPECT_EQ(g.get_adj_vertices(1), v);
    EXPECT_THROW(g.get_adj_vertices(10), std::out_of_range);
}

TEST(GraphTests, GraphTestInsertDelete)
{
    Graph g;
    g.insert_vertex(0);
    g.insert_vertex(1);
    g.insert_vertex(2);
    g.insert_vertex(3);
    EXPECT_EQ(g.num_vertices(), 4);
    EXPECT_EQ(g.num_edges(), 0);

    EXPECT_TRUE(g.has_vertex(2));
    EXPECT_FALSE(g.has_vertex(10));  // no such vertex 10

    EXPECT_FALSE(g.insert_vertex(2)); // already exists

    g.insert_edge(0, 1);
    g.insert_edge(1, 2);
    g.insert_edge(1, 3);
    g.insert_edge(2, 3);
    EXPECT_EQ(g.num_edges(), 4);

    EXPECT_TRUE(g.has_edge(0, 1));  // because graph is undirected
    EXPECT_TRUE(g.has_edge(1, 0));
    EXPECT_TRUE(g.has_edge(2, 3));
    EXPECT_TRUE(g.has_edge(3, 2));
    EXPECT_FALSE(g.has_edge(0, 3));  // no such edges
    EXPECT_FALSE(g.has_edge(2, 2));

    EXPECT_FALSE(g.insert_edge(0, 0));  // loops not allowed
    EXPECT_FALSE(g.insert_edge(0, 5));  // no such vertex 5
    EXPECT_FALSE(g.insert_edge(1, 0));  // edge already exists

    const std::vector<size_t> v = { 0, 2, 3 };
    EXPECT_EQ(g.get_adj_vertices(1), v);
    EXPECT_THROW(g.get_adj_vertices(10), std::out_of_range);

    EXPECT_TRUE(g.delete_edge(2, 1));
    EXPECT_EQ(g.get_adj_vertices(1), std::vector<size_t>({ 0, 3 }));
    EXPECT_EQ(g.num_edges(), 3);

    EXPECT_FALSE(g.delete_edge(5, 1)); // no such edge

    EXPECT_TRUE(g.delete_vertex(1));
    EXPECT_EQ(g.num_vertices(), 3);
    EXPECT_THROW(g.get_adj_vertices(1), std::out_of_range);
    EXPECT_FALSE(g.has_edge(0, 1));  // edge deleted because of vertex 1
    EXPECT_EQ(g.get_adj_vertices(0).size(), 0);
    EXPECT_EQ(g.num_edges(), 1);

    EXPECT_TRUE(g.delete_vertex(0));
    EXPECT_FALSE(g.delete_vertex(1));  // already deleted
    EXPECT_TRUE(g.delete_vertex(2));
    EXPECT_TRUE(g.delete_vertex(3));
    EXPECT_EQ(g.num_vertices(), 0);
}

TEST(GraphTests, GraphTestCustomVertexType)
{
    Graph<uint8_t> g;
    g.insert_vertex(1);
    EXPECT_FALSE(g.insert_vertex(257));  // overflow, unsigned char(257) = 1

}