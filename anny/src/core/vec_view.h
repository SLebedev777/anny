#pragma once

#include "vec.h"


namespace anny
{

/*
    VecView - non-owning linear algebra vector view (or span) for C++ < 20.
*/
template <typename T>
class VecView
{
    static_assert(std::is_arithmetic_v<T>);

public:
    VecView() = default;
    VecView(const VecView&) = default;
    VecView(VecView&&) = default;
    VecView(Vec<T>&&) = delete;  // prohibit creating view from an xvalue

    explicit VecView(Vec<T>& v)
        : m_data(v.m_data.data())
        , m_size(v.size())
    {}

    explicit VecView(const Vec<T>& v)
        : m_data(v.m_data.data())
        , m_size(v.size())
    {}

    VecView(T* data, size_t size)
        : m_data{ data }
        , m_size{ size }
    {}

    template <typename Iter>
    VecView(Iter it, size_t size)
        : m_data(&*it)
        , m_size{ size }
    {}

    template <typename Iter>
    VecView(Iter begin, Iter end)
        : m_data(&*begin)
        , m_size(end - begin)
    {}

    size_t size() const noexcept { return m_size; }
    bool is_same_size(const VecView& other) const noexcept { return size() == other.size(); }

    T& operator[](size_t index) { return m_data[index]; }
    const T& operator[](size_t index) const { return m_data[index]; }

    using iterator = T*;
    using const_iterator = const T*;

    iterator begin() { return iterator(m_data); }
    iterator end() { return iterator(m_data + m_size); }
    const_iterator cbegin() const { return const_iterator(m_data); }
    const_iterator cend() const { return const_iterator(m_data + m_size); }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }


    // math

    VecView& operator+=(T k)
    {
        std::for_each(begin(), end(), [&k](auto& el) { el += k; });
        return *this;
    }

    VecView& operator-=(T k)
    {
        std::for_each(begin(), end(), [&k](auto& el) { el -= k; });
        return *this;
    }

    VecView& operator+=(VecView other)
    {
        assert(is_same_size(other));
        std::transform(begin(), end(), other.begin(), begin(), std::plus<T>());
        return *this;
    }

    VecView& operator-=(VecView other)
    {
        assert(is_same_size(other));
        std::transform(begin(), end(), other.begin(), begin(), std::minus<T>());
        return *this;
    }

    template <typename Const>
    VecView& operator*=(Const k)
    {
        for (size_t i = 0; i < size(); ++i)
            m_data[i] *= k;
        return *this;
    }

    template <typename Const>
    VecView& operator/=(Const k)
    {
        for (size_t i = 0; i < size(); ++i)
            m_data[i] /= k;
        return *this;
    }

    T dot(VecView other) const
    {
        assert(is_same_size(other));
        return std::inner_product(begin(), end(), other.begin(), T{ 0 });
    }

    template <typename DT>
    friend
    DT dot(VecView<DT> left, VecView<DT> right);

private:
    T* m_data = nullptr;
    size_t m_size{0};
};


// math

template <typename T>
Vec<T> operator+(VecView<T> left, VecView<T> right)
{
    Vec<T> result(left);
    result += right;
    return result;
}

template <typename T>
Vec<T> operator-(VecView<T> left, VecView<T> right)
{
    Vec<T> result(left);
    result -= right;
    return result;
}

template <typename T, typename Const>
Vec<T> operator*(VecView<T> v, Const k)
{
    Vec<T> result(v);
    result *= k;
    return result;
}

template <typename T, typename Const>
Vec<T> operator*(Const k, VecView<T> v)
{
    return v * k;
}

template <typename T, typename Const>
Vec<T> operator/(VecView<T> v, Const k)
{
    Vec<T> result(v);
    result /= k;
    return result;
}

template <typename T>
T dot(VecView<T> left, VecView<T> right)
{
    assert(left.is_same_size(right));
    return std::inner_product(left.begin(), left.end(), right.begin(), T{ 0 });
}

template <typename T>
bool operator==(VecView<T> left, VecView<T> right)
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
bool operator!=(VecView<T> left, VecView<T> right)
{
    return !(left == right);
}


}