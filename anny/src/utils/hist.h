#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace anny
{
namespace utils
{

template <typename BinT = double>
struct Bin
{
    BinT from{};
    BinT to{};
    size_t count{};
};


template <typename T, typename BinT = double>
std::vector<Bin<BinT>> build_hist(const std::vector<T>& data, size_t num_bins)
{
    static_assert(std::is_arithmetic<T>::value);

    // init bins
    auto [it_min, it_max] = std::minmax_element(data.begin(), data.end());
    std::vector<Bin<BinT>> hist;
    double bin_width = (*it_max - *it_min) / (1.0 * num_bins);
    double from = *it_min;
    double to = from + bin_width;
    for (size_t i = 0; i < num_bins; i++)
    {
        hist.push_back({ from, to, 0 });
        from += bin_width;
        to += bin_width;
    }

    // fill bins
    for (const auto& x : data)
    {
        for (auto& bin : hist)
        {
            if (x >= bin.from && x < bin.to)
            {
                ++bin.count;
                break;
            }
        }
    }

    return hist;
}


template <typename BinT>
void print_hist(const std::vector<Bin<BinT>> hist, size_t display_width = 60)
{
    // normalize output
    size_t max_count{ 0 };
    for (const auto& bin : hist)
    {
        if (bin.count > max_count)
            max_count = bin.count;
    }
    //print
    for (const auto& bin : hist)
    {
        std::cout << std::setfill(' ');
        std::cout << std::left << "[" << std::setw(7) << bin.from << "; " << std::setw(7) << bin.to << ") <" << bin.count << "> ";
        double ratio = 1.0 * bin.count / (1.0 * max_count);
        int width = static_cast<int>(display_width * ratio);
        for (int i = 0; i < width; i++)
            std::cout << '*';
        std::cout << std::endl;
    }
}

}
}