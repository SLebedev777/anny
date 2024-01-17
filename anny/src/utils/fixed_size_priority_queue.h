#pragma once

#include <queue>

namespace anny
{

template <typename T>
class FixedSizePriorityQueue
{
public:
    FixedSizePriorityQueue() = delete;

    FixedSizePriorityQueue(size_t max_size)
        : m_maxSize(max_size)
    {}

    void push(const T& value)
    {
        if (size() < max_size())
        {
            m_pq.push(value);
            return;
        }
        else
        {
            // T must support operator <
            if (top() < value)
                return;

            m_pq.pop();
            m_pq.push(value);
        }
    }

    const T& top() const { return m_pq.top(); }
    void pop() { m_pq.pop(); }
    bool empty() const { return m_pq.empty(); }
    size_t size() const { return m_pq.size(); }
    size_t max_size() const { return m_maxSize; }

private:
    const size_t m_maxSize;
    std::priority_queue<T> m_pq;  // use only std::less comparer (we need MaxHeap)
};


template <typename T>
std::vector<T> pq2vec(FixedSizePriorityQueue<T>&& pq)
{
    if (pq.empty())
        return {};

    std::vector<T> result(pq.size());
    size_t i = pq.size() - 1;
    while (!pq.empty())
    {
        result[i] = pq.top();
        pq.pop();
        i--;
    }
    return result;
}

}