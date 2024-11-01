# Compiler
CXX = nvcc

# Compiler flags
CXXFLAGS = -std=c++17 -arch=sm_80 -O2 -Xcompiler -Wno-deprecated-declarations

# Linker flags
LDFLAGS = -L$(LIBRARY_PATH) -lcusparse -Xlinker -rpath -Xlinker $(LIBRARY_PATH)

CUDA_PATH ?= /lsc/opt/cuda-11.7
INCLUDE_PATH = $(CUDA_PATH)/include
LIBRARY_PATH = $(CUDA_PATH)/lib64

# Ensure CUDA bin directory is in PATH
ifeq (,$(findstring $(CUDA_PATH)/bin,$(PATH)))
	export PATH := $(CUDA_PATH)/bin:$(PATH)
endif

# Executable name
TARGET = test_program

# Source files
CPP_SOURCES = SM_small.cpp
CU_SOURCES = CSLS_small.cu GS_test.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

# Build all target
all: $(TARGET)

# Link the program
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_PATH) -c $< -o $@

# Compile CUDA source files into object files
%.o: %.cu
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_PATH) -c $< -o $@

# Clean the build
clean:
	rm -f $(OBJECTS) $(TARGET)
