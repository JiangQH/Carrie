#ifndef CARRIE_COMMON_HPP_
#define CARRIE_COMMON_HPP_
namespace carrie {
	enum LayerType
	{
		Convolution = 0,
		MaxPool = 1,
		Connected = 2,
		ReLU = 3,
		ELU = 4,
		PReLU = 5,
		BN = 6,
		Dropout = 7,
		Concat = 8
	};

	enum MemState
	{
		UNINITIALIZED,
		INITIALIZED
	};

	#define DISABLE_COPY_AND_ASSIGN(classname) \
  		classname(const classname&)=delete;\
  		classname& operator=(const classname&) = delete

}
#endif
