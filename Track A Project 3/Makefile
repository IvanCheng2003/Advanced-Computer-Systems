CXX=g++
CXXFLAGS=-O3 -std=c++17 -Wall -Wextra -Wshadow -pthread
LDFLAGS=

all: a4_bench

a4_bench: src/bench.cpp src/hashtable.hpp
	$(CXX) $(CXXFLAGS) -o $@ src/bench.cpp $(LDFLAGS)

clean:
	rm -f a4_bench
