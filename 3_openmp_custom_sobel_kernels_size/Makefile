CXX = g++
CXXFLAGS = -std=c++11 -fopenmp `pkg-config --cflags opencv4`
LDFLAGS = -fopenmp `pkg-config --libs opencv4`
TARGET = sobel_custom_size

all: $(TARGET)

$(TARGET): sobel_custom_size.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) sobel_custom_size.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
