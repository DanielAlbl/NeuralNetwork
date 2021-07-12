#include "Trainer.h"

const int X_DIM = 4, Y_DIM = 3;

int main() {
	Classifier n({X_DIM,50,50,50,50,50,50,50,Y_DIM});
	Trainer t(n);

	t.readX("Data/xtrain.csv");
	t.readY("Data/ytrain.csv");


	t.train(10);
	cout << t.test() << endl;

	return 0;
}
