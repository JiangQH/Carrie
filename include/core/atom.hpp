#ifndef CARRIE_ATOM_HPP_
#define CARRIE_ATOM_HPP_
/**
* atom is the basic item that used in the project
* it is used to store data/params
**/
#include <vector>
#include <memory>
namespace carrie {

template <typename T>
class Atom {
public:
	Atom& operator=(const Atom& atom) = delete;
	Atom(const Atom& atom) = delete;

	Atom(): initialized_(false), allocated_(false), size_(0), reference_count_(0), data_(NULL){}
	

	Atom(const std::vector<int>& shape) {
		create(shape);
	}

	~Atom() {
		release();
	}

	
	void create(std::vector<int>& shape) {
		int size = 1;
		for (int i = 0; i < shape.size(); ++i) {
			int s = shape[i];
			size *= s;
			shape_.push_back(s);
		}
		create(size);
	}

	void release();


private:
	// bottom data pointer
	int size_;
	int reference_count_;
	float* data_;
	std::vector<int> shape_;

	void create(const int& size) {
		data_ = new float[size];
		size_ = size;
		reference_count_ = 1;
	}


}; // end of class Atom 
} // end of namespace carrie
#endif