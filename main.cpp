#include "Trainer.h"

const int X_DIM = 4, Y_DIM = 3;

int main() {
	Classifier n({X_DIM,50,50,50,Y_DIM});
	Trainer t(n);

	t.readTraining("Data/xtrain.csv", "Data/ytrain.csv");
	t.readTesting("Data/xtest.csv", "Data/ytest.csv");

	t.train(10);
	cout << t.test() << endl;
}
