#include <iostream>
#include <gtest/gtest.h>
#include "core/vec.h"
#include "core/vec_view.h"
#include "core/matrix.h"
#include "core/distance.h"

using namespace anny;


TEST(DistanceTests, VectorNormTest)
{
    {
        Vec<int> vi = { 1, 2, 3 };
        EXPECT_EQ(l2_norm_squared(vi.view()), 14);

        Vec<double> vd = { 1.0, 2.0, 3.0 };
        EXPECT_EQ(l2_norm(vd.view()), sqrt(14.0));

        auto vd_normed_expected = vd / sqrt(14.0);
        EXPECT_EQ(l2_normalize(vd.view()), vd_normed_expected);
    }
    {
        Vec<double> vd1 = { 1.0, 2.0, 3.0 };
        Vec<double> vd2 = { 4.0, 5.0, 6.0 };
        double dist_expected = sqrt(9.0 + 9.0 + 9.0);
        EXPECT_EQ(l2_distance(vd1.view(), vd2.view()), dist_expected);
    }
}


