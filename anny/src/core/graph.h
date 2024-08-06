#pragma once

#include <vector>
#include <unordered_map>
#include <limits>
#include <stdexcept>
#include <string>

namespace anny
{
    /*
    * Undirected Graph using adjacent lists.
    */
    template <typename vertex_t = std::size_t>
    class Graph
    {
    public:
        Graph() = default;

        Graph(const std::unordered_map<vertex_t, std::vector<vertex_t>>& adj)
            : m_adj{ adj }
            , m_numEdges{ calc_num_edges() }
        {}

        size_t num_vertices() const noexcept
        { 
            return m_adj.size(); 
        }

        size_t num_edges() const noexcept
        {
            return m_numEdges;
        }

        const std::vector<vertex_t>& get_adj_vertices(vertex_t v) const
        {
            auto it = m_adj.find(v);
            if (it != m_adj.end())
                return it->second;
            else
                throw std::out_of_range("No such vertex: " + std::to_string(v));
        }

        bool has_vertex(vertex_t v) const
        {
            return m_adj.count(v) > 0;
        }

        bool has_edge(vertex_t from, vertex_t to) const
        {
            if (!has_vertex(from) || !has_vertex(to))
                return false;

            for (const auto& u : get_adj_vertices(from))
                if (u == to)
                    return true;
            return false;
        }

        bool insert_vertex(vertex_t v)
        {
            return m_adj.insert({ v, {} }).second;  // map.insert() returns pair <iterator, bool>
        }

        bool insert_edge(vertex_t from, vertex_t to)
        {
            if (from == to)
                return false;  // loops not allowed

            if (!has_vertex(from) || !has_vertex(to))
                return false;

            if (has_edge(from, to))
                return false;

            m_adj[from].push_back(to);
            m_adj[to].push_back(from);
            
            ++m_numEdges;
            return true;
        }

        bool delete_edge(vertex_t from, vertex_t to)
        {
            if (!has_edge(from, to))
                return false;

            m_adj[from].erase(std::remove(m_adj[from].begin(), m_adj[from].end(), to), m_adj[from].end());
            m_adj[to].erase(std::remove(m_adj[to].begin(), m_adj[to].end(), from), m_adj[to].end());

            --m_numEdges;
            return true;
        }

        bool delete_vertex(vertex_t v)
        {
            if (!has_vertex(v))
                return false;

            // need to delete all the adjacent edges
            const auto adj_vertices(get_adj_vertices(v));  // copy
            for (const auto& u : adj_vertices)
            {
                delete_edge(u, v);
            }

            m_adj.erase(v);

            return true;
        }

    private:
        size_t calc_num_edges() const
        {
            std::vector<bool> visited(num_vertices());
            size_t m = 0;
            for (const auto& [u, adj_list]: m_adj)
            {
                for (const auto& v : adj_list)
                {
                    if (!visited[v])
                    {
                        m++;
                    }
                }
                visited[u] = true;
            }
            return m;
        }

    private:
        std::unordered_map<vertex_t, std::vector<vertex_t>> m_adj;
        size_t m_numEdges{ 0 };
    };
}