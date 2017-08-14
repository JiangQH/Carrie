#include "atom.hpp"
#include "util.hpp"
#include <iostream>
namespace carrie {

template <typename T>
Atom<T>::Atom(const std::vector<int>& shape, bool need_diff):
				capacity_(0), need_diff_(need_diff){
	reshape(shape);
}

template <typename T>
void Atom<T>::reshape(const std::vector<int>& shape) {
	shape_.resize(shape.size());
	count_ = 1;
	for (int i = 0; i < shape.size(); ++i) {
		// safety check
		check_larger_than(i, 0);
		shape_[i] = shape[i];
		count_ *= shape[i];
	}
	// space allocation
	if (count_ > capacity_) {
		capacity_ = count_;
		data_.reset(new Mem(sizeof(T) * capacity_));
		if (need_diff_) {
			diff_.reset(new Mem(sizeof(T) * capacity_));
		}
	}
}

template <typename T>
const T* Atom<T>::data() const {
	return (const T*)data_->mem_data();
}

template <typename T>
T* Atom<T>::mutable_data() {
	return (T*)data_->mutable_mem_data();
}

template <typename T>
const T* Atom<T>::diff() const {
	if (!need_diff_) {
		std::cerr << "diff not initialized, return NULL";
		return NULL;
	}
	return (const T*)diff_->mem_data();
}

template <typename T>
T* Atom<T>::mutable_diff() {
	if (!need_diff_) {
		std::cerr << "diff not initialized, return NULL";
		return NULL;
	}
	return (T*)diff_->mutable_mem_data();
}




}