#include <iostream>
#include <gtest/gtest.h>
#include "core/vec.h"
#include "core/vec_view.h"
#include "core/matrix.h"

using namespace anny;


TEST(MatrixTests, MatrixStorageVVTest0)
{
    {
    MatrixStorageVV<int> m1(3, 5);
    EXPECT_EQ(m1.shape().first, 3);
    EXPECT_EQ(m1.shape().second, 5);
    }

    {
    MatrixStorageVV<int> m1 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };
    Shape res_shape{ 3, 3 };
    Vec<int> row = { 4, 5, 6 };
    EXPECT_EQ(m1.shape(), res_shape);
    EXPECT_EQ(m1[1], row.view());
    EXPECT_EQ(m1(2, 0), 7);  // indexing (row, col)
    }
}

TEST(MatrixTests, MatrixCreateTest0)
{
    {
        size_t M = 20, N = 10;
        Matrix<float> m(M, N);
        EXPECT_EQ(m.shape(), Shape(M, N));
    }

    {
        Matrix<int> m = {
            {1, 2, 3},
            {4, 5, 6}
        };
        Shape shape_expected = { 2, 3 };
        EXPECT_EQ(m.shape(), shape_expected);
        Vec<int> row0 = { 1, 2, 3 };
        Vec<int> row1 = { 4, 5, 6 };
        EXPECT_EQ(m[0], row0.view());
        EXPECT_EQ(m[1], row1.view());
    }
}

TEST(MatrixTests, MatrixCalcTest0)
{
    {
        Matrix<int> m = {
            {1, 2, 3},
            {4, 5, 6}
        };
        Matrix<int> expected = {
            {6, 7, 8},
            {9, 10, 11}
        };

        m += 5;
        EXPECT_EQ(m, expected);

    }

    {
        Matrix<int> m = {
            {1, 2, 3},
            {4, 5, 6}
        };
        Matrix<int> expected_same(m);
        Matrix<int> expected_mul = {
            {3, 6, 9},
            {12, 15, 18}
        };

        const int k = 3;
        m *= k;
        EXPECT_EQ(m, expected_mul);
        m /= k;
        EXPECT_EQ(m, expected_same);
    }

    {
        Matrix<int> m = {
            {1, 2, 3},
            {4, 5, 6}
        };
        Matrix<int> expected1 = {
            {0, 0, 0},
            {3, 3, 3}
        };
        Matrix<int> expected2 = {
            {-3, -6, -9},
            {0, -3, -6}
        };

        Vec<int> v = { -1, -2, -3 };
        auto vv = v.view();
        m += vv;

        EXPECT_EQ(m, expected1);

        vv *= -3;  // {3, 6, 9}
        m -= vv;

        EXPECT_EQ(m, expected2);
    }

    {
        Matrix<int> m = {
            {1, 2, 3},
            {4, 5, 6}
        };
        Vec<int> v = { 3, 2, 1 };
        Vec<int> expected_dot = {10, 28};

        auto result = m.dot(v.view());

        EXPECT_EQ(result, expected_dot);

        Vec<int> v_wrong_size = { 1, 2, 3, 4 };
        EXPECT_DEATH(m.dot(v_wrong_size.view()), "Assertion");
    }

}


