SOURCE = main.cpp Matrix.cpp Trainer.cpp Net.cpp Classifier.cpp CNN.cpp CNNTrainer.cpp Tensor.cpp
     
OBJS = $(SOURCE:.cpp=.o)

GCC = g++

LINK = g++

CFLAGS = -Wall -O3 -std=c++17 -I. `pkg-config --cflags --libs opencv4`
CXXFLAGS = $(CFLAGS)

LIBS = `pkg-config  --libs opencv4` 

.PHONY: clean

all : main

main: $(OBJS)
	$(LINK) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o *.d core main

debug: CXXFLAGS = -DDEBUG -g -std=c++17 `pkg-config --cflags --libs opencv4`
debug: main

-include $(SOURCE:.cpp=.d)

%.d: %.cpp
	@set -e; rm -rf $@;$(GCC) -MM $< $(CXXFLAGS) > $@

