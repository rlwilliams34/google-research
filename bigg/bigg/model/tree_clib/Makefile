dir_guard = @mkdir -p $(@D)
FIND := find
CXX := g++

CXXFLAGS += -Wall -O3 -std=c++11
LDFLAGS += -lm 

UNAME := $(shell uname)

CUDA_HOME := /usr/local/cuda-10.2
NVCC := $(CUDA_HOME)/bin/nvcc
USE_GPU = 1

$(info Cuda home = $(CUDA_HOME))
$(info Use GPU = $(USE_GPU))

ifeq ($(UNAME), Darwin)
    USE_GPU = 0
    FOMP := 
else
    LDFLAGS += -fopenmp
    FOMP := -fopenmp
endif

ifeq ($(USE_GPU), 1)
    NVCCFLAGS += --default-stream per-thread
    LDFLAGS += -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand
endif



CUDA_ARCH := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70

build_root = build

ifeq ($(USE_GPU), 1)    
	include_dirs = ./include $(CUDA_HOME)/include
else
	include_dirs = ./include	
endif

$(info include_dirs = $(include_dirs))


CXXFLAGS += $(addprefix -I,$(include_dirs)) -Wno-unused-local-typedef
CXXFLAGS += -fPIC
cpp_files = $(shell $(FIND) src/lib -name "*.cpp" -print | rev | cut -d"/" -f1 | rev)
cxx_obj_files = $(subst .cpp,.o,$(cpp_files))
obj_build_root = $(build_root)/objs
objs = $(addprefix $(obj_build_root)/cxx/,$(cxx_obj_files))


ifeq ($(USE_GPU), 1)
    CXXFLAGS += -DUSE_GPU
    NVCCFLAGS += -DUSE_GPU
    NVCCFLAGS += $(addprefix -I,$(include_dirs))
    NVCCFLAGS += -std=c++11 --use_fast_math --compiler-options '-fPIC'
    cu_files = $(shell $(FIND) src/lib -name "*.cu" -printf "%P\n")
    cu_obj_files = $(subst .cu,.o,$(cu_files))
    objs += $(addprefix $(obj_build_root)/cuda/,$(cu_obj_files))
endif

$(info objs = $(objs))


DEPS = $(objs:.o=.d)

target = $(build_root)/dll/libtree.so
target_dep = $(addsuffix .d,$(target))

$(info target = $(target))
$(info target dep = $(target_dep))

.PRECIOUS: $(build_root)/lib/%.o

$(info dir_guard = $(dir_guard))

all: $(target)

$(target) : src/tree_main.cpp $(objs)
	$(dir_guard)
	$(CXX) -shared $(CXXFLAGS) -MMD -o $@ $(filter %.cpp %.o, $^) $(LDFLAGS)

DEPS += $(target_dep)

$(info obj build root = $(obj_build_root))

ifeq ($(USE_GPU), 1)
$(obj_build_root)/cuda/%.o: src/lib/%.cu
	$(dir_guard)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} -odir $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
endif



$(obj_build_root)/cxx/%.o: src/lib/%.cpp
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $(filter %.cpp, $^) $(FOMP)



clean:
	rm -rf $(build_root)

-include $(DEPS)


