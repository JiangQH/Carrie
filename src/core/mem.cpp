#include "mem.hpp"

namespace carrie {
	Mem(): data_ptr_(NULL), size_(0), state_(UNINITIALIZED), own_data_(false) {}
	Mem(size_t size): data_ptr_(NULL), size_(size), state_(UNINITIALIZED),
					own_data_(false){}
	~Mem() {
		if (data_ptr_ && own_data_) {
			free(data_ptr_);
		}
	}

	// interface
	const void* mem_data() const {
		mem_check();
		return (const void*) data_ptr_;
	}

	void* mutable_mem_data() {
		mem_check();
		return data_ptr_;
	}

	void set_data(void* data) {
		// if we have the data now, free it
		if (data_ptr_ && own_data_) {
			free(data_ptr_);
		}
		data_ptr_ = data;
		state_ = INITIALIZED;
		own_data_ = false;
	}

	void mem_check() {
		switch (state_) {
			case UNINITIALIZED:
				data_ptr_ = malloc(size_);
				memset(data_ptr_, 0, size_);
				state_ = INITIALIZED;
				own_data_ = true;
				break;
			case INITIALIZED:
				break;
		}
	}
}// end of namespace