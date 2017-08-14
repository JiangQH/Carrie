#ifndef CARRIE_ATOM_HPP_
#define CARRIE_ATOM_HPP_
/**
* atom is the basic item that used in the project
* it is used to store data/params
* it's most important func is to wrap the forward data and backward diff together
**/
#include <vector>
#include <memory>
#include "mem.hpp"
namespace carrie {

template <typename T>
class Atom {
public:
	Atom(): diff_(), data_(), count_(0), capacity_(0), need_diff_(false) {}

	explicit Atom(const std::vector<int>& shape, bool need_diff=false);
	void reshape(const std::vector<int>& shape);

	const T* data() const;
	T* mutable_data();
	const T* diff() const;
	T* mutable_diff();

	int count() const {return count_;}
	const std::vector<int>& shape() {return shape_;}
	

private:
	std::shared_ptr<Mem> diff_;
	std::shared_ptr<Mem> data_;
	std::vector<int> shape_;
	int count_;
	int capacity_; // if we need remalloc space
	bool need_diff_;

}; // end of class Atom 
} // end of namespace carrie
#endif