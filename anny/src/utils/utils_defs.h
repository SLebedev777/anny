#ifndef _ANNY_UTILS_DEFS_
#define _ANNY_UTILS_DEFS_

#include "fixed_size_priority_queue.h"
#include "unique_priority_queue.h"

namespace anny
{
namespace utils
{
	template <typename T>
	using UniqueFixedSizePriorityQueue = anny::utils::UniquePriorityQueue<T, anny::utils::FixedSizePriorityQueue<T>>;
}
}
#endif  // _ANNY_UTILS_DEFS_