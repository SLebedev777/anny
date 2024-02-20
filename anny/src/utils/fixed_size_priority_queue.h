#pragma once

#include <queue>


namespace anny
{
namespace utils
{

template <typename T, typename PriorityQueue = std::priority_queue<T>>
class FixedSizePriorityQueue
{
public:
    using value_type = T;

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
    void swap(FixedSizePriorityQueue& other) noexcept
    {
        std::swap(m_maxSize, other.m_maxSize);
        m_pq.swap(other.m_pq);
    }

private:
    size_t m_maxSize;
    PriorityQueue m_pq;  // use only std::less comparer (we need MaxHeap)
};


template <typename PriorityQueue>
std::vector<typename PriorityQueue::value_type> pq2vec(PriorityQueue&& pq)
{
    if (pq.empty())
        return {};

    std::vector<typename PriorityQueue::value_type> result(pq.size());
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
}