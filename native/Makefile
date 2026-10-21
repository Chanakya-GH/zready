.RECIPEPREFIX = >
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
BUILD ?= build

test: $(BUILD)/test_featpack
> $(BUILD)/test_featpack

$(BUILD)/test_featpack: test_featpack.cpp featpack.h
> mkdir -p $(BUILD)
> $(CXX) $(CXXFLAGS) test_featpack.cpp -o $@

clean:
> rm -rf build

.PHONY: test clean
