#ifndef CARRIE_LAYER_HPP_
#define CARRIE_LAYER_HPP_
/**
* this is the basic function item layer
* used to perform specific functions
**/
#include <vector>
#include <string>
#include "atom.hpp"
#include "common.hpp"
namespace carrie {
class Layer {

public:
	virtual void forward(const std::vector<std::vector<Atom>>& bottoms, std::vector<std::vector<Atom>>& tops) = 0;
	
	virtual void load_layer(const std::string& param) = 0;

	virtual void layer_type() const;

	void add_count() {++need_count_;}
	void sub_count() {--need_count_;}
	bool is_clear() { return !need_count_;}

private:
	LayerType type_;
	int need_count_; // count how many layer's input need this layer, since we only need the output

};// end of class
} // end of namespace
#endif