all:
	g++ -O3 -o main main.cpp Matrix.cpp Trainer.cpp Net.cpp classifier.cpp

clean:
	rm -f main 
