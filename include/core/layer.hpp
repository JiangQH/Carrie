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

template <typename T>
class Layer {

public:
	explicit Layer(const std::string& param);
	virtual ~Layer(){}

	virtual void forward(const std::vector<Atom<T>*>& bottoms, std::vector<Atom<T>*>& tops) = 0;
	virtual void backward(const std::vector<Atom<T>*>& top,
		const std::vector<bool>& propagate_down, const std::vector<Atom<T>*>& bottom);
	virtual void load_layer(const std::string& param)=0;
	virtual void layer_type() const;
	
	

private:
	LayerType type_;

};// end of class
} // end of namespace
#endif