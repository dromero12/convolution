CXX = g++
CXXFLAGS = -std=c++11 -fopenmp `pkg-config --cflags opencv4`
LDFLAGS = -fopenmp `pkg-config --libs opencv4`
TARGET = custom_sobel_unit_test

all: $(TARGET)

$(TARGET): custom_sobel_unit_test.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) custom_sobel_unit_test.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
