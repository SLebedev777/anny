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
	class ProgressBar
	{
	public:
		explicit ProgressBar(size_t every_th_iteration)
			: m_n{ every_th_iteration }
		{}

		~ProgressBar()
		{
			std::cout << std::endl << std::flush;
		}

		void update()
		{
			++m_count;
			if (m_count % m_n == 0)
			{
				on_draw();
			}
		}

		void on_draw()
		{
			std::cout << '*' << std::flush;
		}

	private:
		size_t m_n;
		size_t m_count{ 0 };
	};

}
}