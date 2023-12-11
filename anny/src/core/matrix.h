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
    MatrixStorageVV() = delete;  // no sense to create (0, 0) matrix
    MatrixStorageVV(const MatrixStorageVV&) = default;
    MatrixStorageVV(MatrixStorageVV&&) = default;

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

    VecView<DType> operator[](size_t row)  { return m_data[row]; }
    const VecView<DType> operator[](size_t row) const { return m_data[row]; }

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


/*
    Matrix implementation using one whole block of memory. Indexing (i, j) is done as cols*i + j
*/
template <typename DType>
class MatrixStorageContiguous
{
public:
    MatrixStorageContiguous() = delete;
    MatrixStorageContiguous(const MatrixStorageContiguous&) = default;
    MatrixStorageContiguous(MatrixStorageContiguous&&) = default;

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

    Shape shape() const { return { m_rows, m_cols }; }

    VecView<DType> operator[](size_t row) { return VecView<DType>(m_data.data() + row * m_cols, m_cols); }
    const VecView<DType> operator[](size_t row) const { return VecView<DType>(m_data.data() + row * m_cols, m_cols); }

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


/*
    Matrix
*/
template <typename T, typename Storage = MatrixStorageContiguous<T>>
class Matrix
{
public:
    Matrix() = default;

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;

    Matrix(size_t rows, size_t cols)
        : m_storage(rows, cols)
    {}

    Matrix(std::initializer_list<std::initializer_list<T>> lst)
        : m_storage(lst)
    {}

    Shape shape() const { return m_storage.shape(); }

    VecView<T> operator[](size_t row) { return m_storage[row]; }
    const VecView<T> operator[](size_t row) const { return m_storage[row]; }

    T& operator()(size_t row, size_t col) { return m_storage(row, col); }
    const T& operator()(size_t row, size_t col) const { return m_storage(row, col); }

    size_t num_rows() const { return m_storage.num_rows(); }
    size_t num_cols() const { return m_storage.num_cols(); }

private:
    Storage m_storage;
};


}  // namespace anny