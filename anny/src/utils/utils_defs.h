#ifndef _ANNY_UTILS_DEFS_
#define _ANNY_UTILS_DEFS_

#include <limits>
#include <utility>
#include "fixed_size_priority_queue.h"
#include "unique_priority_queue.h"


namespace anny
{
namespace utils
{
	template <typename T>
	using UniqueFixedSizePriorityQueue = anny::utils::UniquePriorityQueue<T, anny::utils::FixedSizePriorityQueue<T>>;

	template <typename T, typename Index>
	using DistIndexPair = std::pair<T, Index>;

	constexpr unsigned int UNDEFINED_SEED = std::numeric_limits<unsigned int>::max();
}
}
#endif  // _ANNY_UTILS_DEFS_