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


void remove_space(char* str) {
	if (str == NULL) {
		return;
	}
	int real = 0;
	int walker = 0;
	while (str[walker] != '/0') {
		if (walker == real && str[walker] != ' ') {
			++walker;
			++real;
		}
		else if (str[walker] == ' ') {
			++walker;
		}
		else {
			str[real++] = str[walker++];
		}
	}
	str[real] = '/0';
}