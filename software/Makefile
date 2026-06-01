# -------- Compiler & flags --------
CXX := g++

CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -pedantic \
            -fopenmp \
            -I../

# -------- Target --------
TARGET := beamsearch

# -------- Sources --------
SRCS := \
	main.cpp \
	utils.cpp \
	globals.cpp \
	Instance.cpp \
	nnet_parallel.cpp

OBJS := $(SRCS:.cpp=.o)

# -------- Default rule --------
all: $(TARGET)

# -------- Link --------
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^

# -------- Compile --------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

# -------- Clean --------
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
