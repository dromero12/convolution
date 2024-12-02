CXX = g++
CXXFLAGS = -std=c++11 -fopenmp `pkg-config --cflags opencv4`
LDFLAGS = -fopenmp `pkg-config --libs opencv4`
TARGET = sobel_openmp

all: $(TARGET)

$(TARGET): sobel_openmp.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) sobel_openmp.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
