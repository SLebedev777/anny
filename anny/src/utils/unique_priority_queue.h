#pragma once

#include <queue>
#include <set>


namespace anny
{
namespace utils
{

/* Priority queue that contains only unique elements.
*/

template<
    class T,
    class PriorityQueue = std::priority_queue<T>> 
class UniquePriorityQueue: public PriorityQueue  // CRTP to automatically inherit base class interface
{
public:
    using value_type = T;

    // wrap rvalue base class object
    UniquePriorityQueue(PriorityQueue&& pq)
        : PriorityQueue(std::move(pq))
    {}

    void push(const T& value)
    {
        if (m_unique.find(value) == m_unique.end())
        {
            m_unique.insert(value);
            PriorityQueue::push(value);
        }
    }

    const T& top() const { return PriorityQueue::top(); }
    
    void pop() 
    {
        auto it = m_unique.find(top());
        m_unique.erase(it);
        PriorityQueue::pop();
    }
    
    bool empty() const { return PriorityQueue::empty(); }
    size_t size() const { return PriorityQueue::size(); }
    void swap(UniquePriorityQueue& other) noexcept
    {
        PriorityQueue::swap(other);
        m_unique.swap(other.m_unique);
    }

private:
    std::set<T> m_unique;
};

}
}