#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <type_traits>


namespace anny
{

inline constexpr size_t END = std::numeric_limits<size_t>::max();


template <typename T> class VecView;

/*
    Vec - linear algebra vector of numbers, dynamically allocated and owning its memory.
*/
template <typename T>
class Vec
{
    static_assert(std::is_arithmetic_v<T>);

public:
    template <typename DT>
    friend
    class VecView;

    using value_type = std::remove_cv_t<T>;  // to be able to create std::vector<T> from VecView<const T>


    Vec() = default;

    explicit Vec(size_t size, T v = T{})
    : m_data(size, v)
    {}

    Vec(std::initializer_list<T> lst)
    : m_data(lst)
    {}

    Vec(const std::vector<T>& data)
    : m_data(data)
    {}

    explicit Vec(VecView<T> vv)
    : m_data(vv.begin(), vv.end())
    {}

    explicit Vec(const VecView<const T>& vv)
        : m_data(vv.begin(), vv.end())
    {}

    Vec(const Vec& other)
    : m_data(other.m_data)
    {}

    Vec(Vec&& other)
    : m_data(std::move(other.m_data))
    {}

    Vec& operator=(const Vec& other)
    {
        m_data = other.m_data;
        return *this;
    }

    Vec& operator=(Vec&& other) noexcept
    {
        m_data = std::move(other.m_data);
        return *this;
    }

    size_t size() const noexcept { return m_data.size(); }
    bool is_same_size(const Vec& other) const noexcept  { return size() == other.size(); }
    bool is_same_size(VecView<T> other) const noexcept { return size() == other.size(); }

    value_type& operator[](size_t index) { return m_data[index]; }
    const value_type& operator[](size_t index) const { return m_data[index]; }

    VecView<value_type> view() { return VecView<value_type>(*this); }
    VecView<const value_type> view() const { return VecView<const value_type>(m_data.begin(), m_data.end()); }
    VecView<value_type> view(size_t start, size_t size = END)
    {
        auto iter_start = m_data.begin() + start;
        auto iter_end = (size != END) ? (iter_start + size) : m_data.end();
        return VecView<value_type>(iter_start, iter_end);
    }

    // math

    Vec& operator+=(T k)
    {
        std::for_each(m_data.begin(), m_data.end(), [&k](auto& el) { el += k; });
        return *this;
    }

    Vec& operator-=(T k)
    {
        std::for_each(m_data.begin(), m_data.end(), [&k](auto& el) { el -= k; });
        return *this;
    }

    Vec& operator+=(const Vec& other)
    {
        assert(is_same_size(other));
        std::transform(m_data.begin(), m_data.end(), other.m_data.begin(), m_data.begin(), std::plus<T>());
        return *this;
    }

    Vec& operator-=(const Vec& other)
    {
        assert(is_same_size(other));
        std::transform(m_data.begin(), m_data.end(), other.m_data.begin(), m_data.begin(), std::minus<T>());
        return *this;
    }

    Vec& operator+=(VecView<T> other)
    {
        assert(is_same_size(other));
        std::transform(m_data.begin(), m_data.end(), other.begin(), m_data.begin(), std::plus<T>());
        return *this;
    }

    Vec& operator-=(VecView<T> other)
    {
        assert(is_same_size(other));
        std::transform(m_data.begin(), m_data.end(), other.begin(), m_data.begin(), std::minus<T>());
        return *this;
    }

    template <typename Const>
    Vec& operator*=(Const k)
    {
        for (size_t i = 0; i < size(); ++i)
            m_data[i] *= k;
        return *this;
    }

    template <typename Const>
    Vec& operator/=(Const k)
    {
        for (size_t i = 0; i < size(); ++i)
            m_data[i] /= k;
        return *this;
    }

    T dot(const Vec& other) const
    {
        assert(is_same_size(other));
        return std::inner_product(m_data.begin(), m_data.end(), other.m_data.begin(), T{0});
    }

    template <typename DT>
    friend
    DT dot(const Vec<DT>& left, const Vec<DT>& right);

private:
    std::vector<Vec::value_type> m_data;
};


template <typename T>
Vec<T> operator+(const Vec<T>& left, const Vec<T>& right)
{
    Vec<T> result{left};
    result += right;
    return result;
}

template <typename T>
Vec<T> operator-(const Vec<T>& left, const Vec<T>& right)
{
    Vec<T> result{left};
    result -= right;
    return result;
}

template <typename T, typename Const>
Vec<T> operator*(const Vec<T>& v, Const k)
{
    Vec<T> result{v};
    result *= k;
    return result;
}

template <typename T, typename Const>
Vec<T> operator*(Const k, const Vec<T>& v)
{
    return v * k;
}

template <typename T, typename Const>
Vec<T> operator/(const Vec<T>& v, Const k)
{
    Vec<T> result{v};
    result /= k;
    return result;
}

template <typename T>
T dot(const Vec<T>& left, const Vec<T>& right)
{
    assert(left.is_same_size(right));
    return std::inner_product(left.m_data.begin(), left.m_data.end(), right.m_data.begin(), T{0});
}

template <typename T>
bool operator==(const Vec<T>& left, const Vec<T>& right)
{
    if (!left.is_same_size(right))
        return false;

    for (size_t i = 0; i < left.size(); ++i)
    {
        if (left[i] != right[i])
            return false;
    }
    return true;
}

template <typename T>
bool operator!=(const Vec<T>& left, const Vec<T>& right)
{
    return !(left == right);
}

}  // namespace anny