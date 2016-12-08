#ifndef CARRIE_CHECK_HPP_
#define CARRIE_CHECK_HPP_
#include <stdexcept>
void CARRIE_CHECK_EQ(bool is_equal) {
    if (!is_equal) {
        throw runtime_error("Carrie Error: dimension not equal error");
    }
}

void CARRIE_CHECK_GT(bool is_larger) {
    if (!is_larger) {
        throw runtime_error("Carrie Error: should be larger");
    }
}
#endif
