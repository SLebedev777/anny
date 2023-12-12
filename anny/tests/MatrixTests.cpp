#include <iostream>
#include <gtest/gtest.h>
#include "core/vec.h"
#include "core/vec_view.h"
#include "core/matrix.h"

using namespace anny;

TEST(VecTests, VecCalcTest0)
{
    {
    Vec<int> v1{1, 2, 3};
    Vec<int> v2{3, 4, 5};

    EXPECT_EQ(v1.dot(v2), 26);
    }

    {
    Vec<int> v1{1, 2, 3};
    Vec<int> result{10, 20, 30};

    EXPECT_EQ(v1 * 10, result);
    EXPECT_EQ(10 * v1, result);
    }

    {
    Vec<double> v1{1, 2, 3};

    EXPECT_EQ(v1 / 2, Vec<double>({0.5, 1.0, 1.5}));
    }

    {
    Vec<int> v1{ 1, 2, 3 };
    Vec<int> expected_minus(v1);
    Vec<int> expected_plus{ 6, 7, 8 };
    v1 += 5;
    EXPECT_EQ(v1, expected_plus);
    v1 -= 3;
    v1 -= 2;
    EXPECT_EQ(v1, expected_minus);
    }
}

TEST(VecViewTests, VecViewCtorTest0)
{
    {
    Vec<int> v1{1, 2, 3};
    VecView<int> vv1(v1);
    auto vv2 = v1.view();
    EXPECT_EQ(vv1, vv2);
    EXPECT_EQ(vv1, v1.view());
    
    Vec<int> v2{ 3, 4 };
    EXPECT_NE(v2.view(), v1.view());
    
    }
}

TEST(VecViewTests, VecViewIterTest0)
{
    {
        Vec<int> v1{ 1, 2, 3 };
        for (auto x : v1.view())
            std::cout << x << " ";

        auto vv = v1.view();
        for (auto it = vv.begin(); it != vv.end(); ++it)
            *it *= 2;

        Vec<int> expected{ 2, 4, 6 };
        EXPECT_EQ(v1, expected);
    }
    {
        Vec<int> v1{ 1, 2, 3 };
        auto vv = v1.view();
        VecView<int> vv2(vv.begin() + 1, 2);  // {2, 3}
        EXPECT_EQ(vv2[0], 2);
        EXPECT_EQ(vv2[1], 3);
        EXPECT_EQ(vv2.size(), 2);
    }
}

TEST(VecViewTests, VecViewCalcTest0)
{
    {
        Vec<int> v1{ 1, 2, 3 };
        Vec<int> v2{ 4, 6, 8 };
        auto vv1 = v1.view();
        auto vv2 = v2.view();
        vv1 *= 4;
        vv2 /= 2;
        Vec<int> res(vv1 - vv2);
        Vec<int> expected{ 2, 5, 8 };
        EXPECT_EQ(res, expected);
    }
    {
        Vec<int> v1{ 1, 2, 3, 4, 5, 6 };
        Vec<int> v2{ 6, 5, 4, 3, 2, 1 };
        auto vv1 = v1.view(3, 3);  // {4, 5, 6}
        auto vv2 = v2.view(3);     // {3, 2, 1}
        EXPECT_EQ(dot(vv1, vv2), 28);
    }
    {
        Vec<int> v1{ 1, 2, 3 };
        Vec<int> expected_minus(v1);
        Vec<int> expected_plus{ 6, 7, 8 };
        auto vv1 = v1.view();
        vv1 += 5;
        EXPECT_EQ(vv1, expected_plus.view());
        vv1 -= 3;
        vv1 -= 2;
        EXPECT_EQ(vv1, expected_minus.view());
    }

}

TEST(VecViewTests, VecViewConstnessTest0)
{}

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
}

