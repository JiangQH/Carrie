#ifndef CARRIE_ATOM_HPP_
#define CARRIE_ATOM_HPP_
/**
* atom is the basic item that used in the project
* it is used to store data/params
**/
#include <vector>
#include <memory>
namespace carrie {
class Atom {
public:
	
	Atom();
	Atom(const std::vector<int> shape);
	~Atom();

	void create(const int& size);

	// shadow copy
	Atom(const Atom& atom);
	Atom& operator=(const Atom& atom);

	// deep copy
	Atom deep_copy(const Atom& atom);

private:
	// bottom data pointer
	bool initialized_;
	bool allocated_;
	std::shared_ptr<float> data_ptr_; 
	std::vector<int> shape_;

	template< typename T >
	struct array_deleter{
		void operator ()( T const * p){ 
    		delete[] p; 
  		}
	};


}; // end of class Atom
} // end of namespace carrie
#endif