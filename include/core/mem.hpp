#ifndef CARRIE_MEM_HPP_
#define CARRIE_MEM_HPP_
/**
* memory interface to outside, ideas bowrow from caffe
**/
#include "common.hpp"
namespace carrie {
class Mem {
public:
	Mem();
	explicit Mem(size_t size);
	~Mem();

	// interface
	const void* mem_data() const;
	void* mutable_mem_data();
	void set_data(void* data);
	size_t size() const { return size_; }

private:

	void* data_ptr_;
	size_t size_;
	MemState state_;
	bool own_data_;

	// to make sure only allocate space when we need it
	void mem_check();
	DISABLE_COPY_AND_ASSIGN(Mem);
};
}// end of namespace
#endif