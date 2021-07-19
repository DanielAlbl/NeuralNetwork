#include "Trainer.h"
#include "CNN.h"

const int X_DIM = 4, Y_DIM = 3;

int main() {
	CNN cnn({ 64,64,3 }, { 12,0,10,0,5,0 }, { 50,50,9 });


	//Classifier n({X_DIM,50,50,50,Y_DIM});
	//Trainer t(n);

	//t.readTraining("Data/xtrain.csv", "Data/ytrain.csv");
	//t.readTesting("Data/xtest.csv", "Data/ytest.csv");

	//t.train(10);
	//cout << t.test() << endl;
}
