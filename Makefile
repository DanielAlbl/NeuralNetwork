SOURCE = main.cpp Matrix.cpp Trainer.cpp Net.cpp Classifier.cpp
     
OBJS = $(SOURCE:.cpp=.o)

GCC = g++

LINK = g++

CFLAGS = -Wall -O3 -std=c++11 -I. 
CXXFLAGS = $(CFLAGS)

LIBS = 

.PHONY: clean

all : main

main: $(OBJS)
	$(LINK) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o *.d core main

debug: CXXFLAGS = -DDEBUG -g -std=c++11
debug: main

-include $(SOURCE:.cpp=.d)

%.d: %.cpp
	@set -e; rm -rf $@;$(GCC) -MM $< $(CXXFLAGS) > $@

