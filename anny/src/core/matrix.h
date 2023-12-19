#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include "vec.h"


namespace anny
{
    using Shape = std::pair<size_t, size_t>;

/*
    MatrixStorageVV - matrix storage implementation where each row is a separate vector
*/
template <typename DType>
class MatrixStorageVV
{
public:
    MatrixStorageVV() = default;
    MatrixStorageVV(const MatrixStorageVV&) = default;
    MatrixStorageVV(MatrixStorageVV&&) = default;
    MatrixStorageVV& operator=(MatrixStorageVV&&) = default;

    MatrixStorageVV(size_t rows, size_t cols)
    {
        assert(rows > 0 && cols > 0);

        for (size_t i = 0; i < rows; ++i)
            m_data.push_back(std::move(Vec<DType>(cols)));
    }

    MatrixStorageVV(std::initializer_list<std::initializer_list<DType>> lst)
    : m_data(lst.size())
    {
        // TODO: check lst is not empty
        size_t i = 0;
        for (const auto& row: lst)
        {
            m_data[i] = row;
            ++i;
        }
    }

    // copy ctor from data stored in STL container
    MatrixStorageVV(const std::vector<std::vector<DType>>& data)
    {
        m_data.reserve(data.size());
        for (const auto& row : data)
            m_data.push_back(row);
    }

    // move ctor from data stored in STL container
    MatrixStorageVV(std::vector<std::vector<DType>>&& data)
        : m_data(std::move(data))
    {
        m_data.reserve(data.size());
        for (auto&& row : data)
            m_data.push_back(std::move(row));
    }


    VecView<DType> operator[](size_t row)  { return m_data[row]; }
    VecView<const DType> operator[](size_t row) const { return m_data[row]; }

    DType& operator()(size_t row, size_t col)  { return m_data[row][col]; }
    const DType& operator()(size_t row, size_t col) const  { return m_data[row][col]; }

    Shape shape() const { return {num_rows(), num_cols()}; }
    size_t num_rows() const { return m_data.size(); }
    size_t num_cols() const { return m_data[0].size(); }

    void add_row(const Vec<DType>& v)
    {
        assert(m_data[0].is_same_size(v));
        m_data.push_back(v);
    }

private:
    std::vector<Vec<DType>> m_data;
};

template <typename T>
bool operator==(const MatrixStorageVV<T>& left, const MatrixStorageVV<T>& right)
{
    if (left.shape() != right.shape())
        return false;
    for (size_t i = 0; i < left.num_rows(); i++)
    {
        if (left[i] != right[i])
            return false;
    }
    return true;
}

template <typename T>
bool operator!=(const MatrixStorageVV<T>& left, const MatrixStorageVV<T>& right)
{
    return !(left == right);
}


/*
    Matrix implementation using one whole block of memory. Indexing (i, j) is done as cols*i + j
*/
template <typename DType>
class MatrixStorageContiguous
{
public:
    MatrixStorageContiguous() = default;
    MatrixStorageContiguous(const MatrixStorageContiguous&) = default;
    MatrixStorageContiguous(MatrixStorageContiguous&&) = default;
    MatrixStorageContiguous& operator=(MatrixStorageContiguous&&) = default;

    MatrixStorageContiguous(size_t rows, size_t cols)
    : m_data(rows * cols)
    , m_rows{rows}
    , m_cols{cols}
    {}

    MatrixStorageContiguous(std::initializer_list<std::initializer_list<DType>> lst)
    : m_data(lst.size() * lst.begin()->size())
    , m_rows{lst.size()}
    , m_cols{ lst.begin()->size() }
    {
        // TODO: check lst is not empty
        auto iter = m_data.begin();
        for (const auto& row : lst)
        {
            std::copy(row.begin(), row.end(), iter);
            iter += m_cols;
        }
    }

    MatrixStorageContiguous(const std::vector<DType>& data, size_t cols)
        : m_data(data)
        , m_rows(data.size())
        , m_cols(cols)
    {}

    MatrixStorageContiguous(std::vector<DType>&& data, size_t cols)
        : m_data(std::move(data))
        , m_rows(data.size())
        , m_cols(cols)
    {}

    Shape shape() const { return { m_rows, m_cols }; }

    VecView<DType> operator[](size_t row) { return VecView<DType>(m_data.data() + row * m_cols, m_cols); }
    VecView<const DType> operator[](size_t row) const { return VecView<const DType>(m_data.data() + row * m_cols, m_cols); }

    DType& operator()(size_t row, size_t col) { return m_data[pos(row, col)]; }
    const DType& operator()(size_t row, size_t col) const { return m_data[pos(row, col)]; }

    size_t num_rows() const { return m_rows; }
    size_t num_cols() const { return m_cols; }

private:
    inline size_t pos(size_t row, size_t col) const noexcept { return row * m_cols + col; }

private:
    std::vector<DType> m_data;
    size_t m_rows;
    size_t m_cols;
};


template <typename T>
bool operator==(const MatrixStorageContiguous<T>& left, const MatrixStorageContiguous<T>& right)
{
    if (left.shape() != right.shape())
        return false;
    for (size_t i = 0; i < left.num_rows(); i++)
    {
        if (left[i] != right[i])
            return false;
    }
    return true;
}

template <typename T>
bool operator!=(const MatrixStorageContiguous<T>& left, const MatrixStorageContiguous<T>& right)
{
    return !(left == right);
}



/*
    Matrix
*/
template <typename T, typename Storage = MatrixStorageContiguous<T>>
class Matrix
{
    static_assert(std::is_arithmetic_v<T>);

public:
    Matrix() = default;

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&& other) = default;

    Matrix(size_t rows, size_t cols)
        : m_storage(rows, cols)
    {}

    Matrix(std::initializer_list<std::initializer_list<T>> lst)
        : m_storage(lst)
    {}

    Matrix(const Storage& storage)
        : m_storage(storage)
    {}

    Shape shape() const { return m_storage.shape(); }

    VecView<T> operator[](size_t row) { return m_storage[row]; }
    const VecView<T> operator[](size_t row) const { return m_storage[row]; }

    T& operator()(size_t row, size_t col) { return m_storage(row, col); }
    const T& operator()(size_t row, size_t col) const { return m_storage(row, col); }

    size_t num_rows() const { return m_storage.num_rows(); }
    size_t num_cols() const { return m_storage.num_cols(); }

    // math

    // operations with a number

    Matrix& operator+=(const T& k)
    {
        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            row_view += k;
        }
        return *this;
    }

    Matrix& operator-=(const T& k)
    {
        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            row_view -= k;
        }
        return *this;
    }

    Matrix& operator*=(const T& k)
    {
        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            row_view *= k;
        }
        return *this;
    }

    Matrix& operator/=(const T& k)
    {
        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            row_view /= k;
        }
        return *this;
    }

    // operations with a vector

    Matrix& operator+=(const VecView<T>& vec)
    {
        assert(vec.size() == num_cols());

        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            row_view += vec;
        }
        return *this;
    }

    Matrix& operator-=(const VecView<T>& vec)
    {
        assert(vec.size() == num_cols());

        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            row_view -= vec;
        }
        return *this;
    }

    Vec<T> dot(const VecView<T>& v)
    {
        Vec<T> result(num_rows());
        for (size_t i = 0; i < num_rows(); ++i)
        {
            auto row_view = m_storage[i];
            result[i] = anny::dot(row_view, v);
        }
        return result;
    }

    template <typename _T, typename _Storage>
    friend
    bool operator==(const Matrix<_T, _Storage>& left, const Matrix<_T, _Storage>& right);

    template <typename _T, typename _Storage>
    friend
    bool operator!=(const Matrix<_T, _Storage>& left, const Matrix<_T, _Storage>& right);

private:
    Storage m_storage;
};


template <typename T, typename Storage>
bool operator==(const Matrix<T, Storage>& left, const Matrix<T, Storage>& right)
{
    return left.m_storage == right.m_storage;
}

template <typename T, typename Storage = MatrixStorageContiguous<T>>
bool operator!=(const Matrix<T, Storage>& left, const Matrix<T, Storage>& right)
{
    return !(left == right);
}


}  // namespace anny