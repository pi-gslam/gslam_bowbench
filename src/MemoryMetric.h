#include <GSLAM/core/SharedLibrary.h>
#include <unordered_map>

#ifdef ANDROID
extern "C" {
void *malloc(size_t size) ;
void free(void* ptr) ;
}
#else
extern "C" {
void *malloc(size_t size) throw();
void free(void* ptr) throw();
}
#endif

namespace GSLAM{

class MemoryMetric{
public:
    MemoryMetric():_usage(0),_enabled(false),_shouldIgnore(false){}
    ~MemoryMetric(){_enabled=false;}

    static MemoryMetric& instanceCPU(){
        static MemoryMetric inst;
        return inst;
    }

    bool isEnabled()const{return _enabled;}

    void enable(){_enabled=true;}

    size_t usage()const{return _usage;}
    size_t count()const{return _allocated_sizes.size();}

    void AddAllocation(void* ptr,size_t size){
        if(!_enabled) return;
        if(_shouldIgnore) return;

        {
            std::unique_lock<std::mutex> lock(_mutex);
            _shouldIgnore=true;
            _allocated_sizes[ptr] = size;
            _usage+=size;
            _shouldIgnore=false;
        }
    }

    void FreeAllocation(void* ptr){
        if(!_enabled) return;
        if(_shouldIgnore) return;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _shouldIgnore=true;
            auto it=_allocated_sizes.find(ptr);
            if(it!=_allocated_sizes.end())
            {
                _usage-=it->second;
                _allocated_sizes.erase(it);
            }
            _shouldIgnore=false;
        }
    }

    operator bool(){return isEnabled();}

private:
    std::map<void*,size_t>  _allocated_sizes;
    std::mutex _mutex;
    size_t     _usage;
    bool       _enabled,_shouldIgnore;
};

}
