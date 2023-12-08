#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include "vec.h"

namespace anny
{

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

    Vec<DType>& operator[](size_t row)  { return m_data[row]; }
    const Vec<DType>& operator[](size_t row) const { return m_data[row]; }

    DType& operator()(size_t row, size_t col)  { return m_data[row][col]; }
    const DType& operator()(size_t row, size_t col) const  { return m_data[row][col]; }

    std::pair<size_t, size_t> shape() const { return {num_rows(), num_cols()}; }
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
    : m_data(lst.size() * lst[0].size())
    , m_rows{lst.size()}
    , m_cols{ lst[0].size() }
    {
        // TODO: check lst is not empty
        size_t i = 0;
        for (const auto& row : lst)
        {
            m_data[i] = row;
            ++i;
        }
    }

    std::pair<size_t, size_t> shape() const { return { m_rows, m_cols }; }

private:
    inline size_t pos(size_t row, size_t col) const noexcept { return row * m_rows + col; }

private:
    std::vector<DType> m_data;
    size_t m_rows;
    size_t m_cols;
};


/*
    Matrix
*/

}  // namespace anny