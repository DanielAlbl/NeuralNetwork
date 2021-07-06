all:
	g++ -fopenmp -o main main.cpp Matrix.cpp Trainer.cpp Net.cpp

clean:
	rm -f main 
