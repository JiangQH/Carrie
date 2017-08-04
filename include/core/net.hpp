#ifndef CARRIE_NET_HPP_
#define CARRIE_NET_HPP_
/**
* this is the net structure, used to hold the layer function items
* the optimizer will query this structure to get the actual function item
**/
#include "layer.hpp"
#include "atom.hpp"
#include <string>
#include <vector>
#include <memory>

namespace carrie {
class Net{
public:
	void build_net(const std::string& str);
	void forward();

private:
	// store the layers
	std::vector<std::shared_ptr<Layer>> layers_;
	std::vector<std::string> layer_names_;

	// hold the data, until we no longer need it
	std::vector<Atom> datas_;
	std::vector<int> data_ids_;


}; //end of carrie
} // end of namespace
#endif