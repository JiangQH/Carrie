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
	
	
private:
	std::shared_ptr<Mem> diff_;
	std::shared_ptr<Mem> data_;
	std::vector<int> shape_;
	int count_;
	int capacity_; // if we need remalloc space


}; // end of class Atom 
} // end of namespace carrie
#endif