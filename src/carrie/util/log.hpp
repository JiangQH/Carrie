#ifndef CARRIE_LOG_H_
#define CARRIE_LOG_H_
#include <iostream>
#include <string>
enum logtype {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class LOG {
public:
    LOG() {}
    LOG(logtype type) {
        msglevel_ = type;
    }
    ~LOG() {
        if(opened_) {
            std::cout << std::endl;
        }
        opened_ = false;
    }

    template<class T>
    LOG& operator<<(const T& msg) {
        std::cout << "[" + get_lable() + "]" <<": " <<msg;
        opened_ = true;
        return *this;
    }

private:
    bool opened_ = false;
    logtype msglevel_ = DEBUG;
    std::string get_lable() const {
        std::string label;
        switch(msglevel_) {
            case DEBUG: label = "Debug"; break;
            case INFO: label = "Info"; break;
            case WARN: label = "Warning"; break;
            case ERROR: label = "Error"; break;
        }
        return label;
    } 
};    
#endif
