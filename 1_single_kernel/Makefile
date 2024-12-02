CXX = g++
CXXFLAGS = -std=c++11 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = single_kernel

all: $(TARGET)

$(TARGET): single_kernel.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) single_kernel.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
