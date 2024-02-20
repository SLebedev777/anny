#include <iostream>
#include <gtest/gtest.h>
#include "core/vec.h"
#include "core/vec_view.h"
#include "core/hyperplane.h"


using namespace anny;

TEST(HyperplaneTests, HyperplaneDefaultCtorTest)
{
    {
        Hyperplane<float> h;

        EXPECT_EQ(h.normal.size(), 0);
        EXPECT_EQ(h.intercept, 0.0f);
        Vec<float> vec_zero{};
        EXPECT_EQ(h.distance(vec_zero.view()), 0);
    }
}

TEST(HyperplaneTests, HyperplaneDistanceTest)
{
    {
        // this vertical 2d plane has equation x - 5 = 0 for every y
        double intercept = -5.0;
        auto h = make_orthogonal_hyperplane<double>(2, 0, intercept);

        double delta = 3.0;
        Vec<double> point{ -intercept + delta, 3 };  // (8;3)

        EXPECT_EQ(h.distance(point.view()), delta);
    }

    {
        const float c = 12345.12345f;
        Vec<float> n{ -c, c };  // 45 degrees diagonal 2d line
        n = l2_normalize(n.view());

        float intercept = -sqrt(2.0f);  // so the plane goes through points (-2; 0) and (0; 2)
        Hyperplane<float> h(n, intercept);

        Vec<float> point1{ -2.0f, 0.0f };
        Vec<float> point2{ 0.0f, 2.0f };
        Vec<float> point3{ 1.0f, 6.0f }; // 2 points equidistant from plane
        Vec<float> point4{ 4.0f, 3.0f };

        auto d1 = h.distance(point1.view());
        auto d2 = h.distance(point2.view());
        auto d3 = h.distance(point3.view());
        auto d4 = h.distance(point4.view());

        EXPECT_TRUE(anny::are_floats_equal(d1, 0.0f));
        EXPECT_TRUE(anny::are_floats_equal(d2, 0.0f));
        EXPECT_TRUE(anny::are_floats_equal(d3, d4));

    }
}

TEST(HyperplaneTests, HyperplaneSideTest)
{
    {
        const float c = 12345.12345f;
        Vec<float> n{ -c, c };  // 45 degrees diagonal 2d line
        n = l2_normalize(n.view());

        float intercept = -sqrt(2.0f);  // so the plane goes through points (-2; 0) and (0; 2)
        Hyperplane<float> h(n, intercept);

        Vec<float> point3{ 1.0f, 6.0f }; // 2 points equidistant from plane
        Vec<float> point4{ 4.0f, 3.0f };

        EXPECT_TRUE(h.side(point3.view()));  // point above plane 
        EXPECT_FALSE(h.side(point4.view())); // point below plane

    }

}

TEST(HyperplaneTests, HyperplaneThroughGivenPointTest)
{
    {
        const double c = 1.0;
        Vec<double> n{ -c, c };  // 45 degrees diagonal 2d line
        n = l2_normalize(n.view());
        Vec<double> x0{ 2.5, 4.5 };

        Hyperplane<double> h(n, x0);

        Vec<double> point1{ 0.0, 2.0 };
        Vec<double> point3{ 1.0, 6.0 }; // 2 points equidistant from plane
        Vec<double> point4{ 4.0, 3.0 };

        auto d1 = h.distance(point1.view());

        EXPECT_TRUE(anny::are_floats_equal(h.intercept, -sqrt(2.0)));
        EXPECT_TRUE(anny::are_floats_equal(d1, 0.0));
        std::cout << d1 << std::endl;
        std::cout << "[" << h.normal[0] << ", " << h.normal[1] << std::endl;
    }

    {
        const float c = 1.0f;
        Vec<float> n{ -c, c };  // 45 degrees diagonal 2d line
        n = l2_normalize(n.view());
        Vec<float> x0{ 2.5f, 4.5f };

        Hyperplane<float> h(n, x0);

        Vec<float> point1{ 0.0f, 2.0f };
        Vec<float> point3{ 1.0f, 6.0f }; // 2 points equidistant from plane
        Vec<float> point4{ 4.0f, 3.0f };

        auto d1 = h.distance(point1.view());

        EXPECT_TRUE(anny::are_floats_equal(h.intercept, -sqrtf(2.0f)));
        EXPECT_TRUE(anny::are_floats_equal(d1, 0.0f));
        std::cout << d1 << std::endl;
        std::cout << "[" << h.normal[0] << ", " << h.normal[1] << std::endl;
    }

}