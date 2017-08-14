#include <iostream>

namespace carrie {
	template <typename T>
	void check_larger_than(const T& a, const T& b) {
		if (a <= b) {
			std::cerr << "Larger than not satisfied " << a << " vs " << b;
			exit(1);
		}
	}
	
}