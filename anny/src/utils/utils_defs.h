#ifndef _ANNY_UTILS_DEFS_
#define _ANNY_UTILS_DEFS_

#include "fixed_size_priority_queue.h"
#include "unique_priority_queue.h"

namespace anny
{
	template <typename T>
	using UniqueFixedSizePriorityQueue = anny::UniquePriorityQueue<T, anny::FixedSizePriorityQueue<T>>;
}

#endif  // _ANNY_UTILS_DEFS_