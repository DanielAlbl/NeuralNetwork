#include "Trainer.h"
#include "CNNTrainer.h"

int main() {
	CNN cnn({ 28,28,1 }, { 16,0,16,0,16 }, { 50,50,10 });
	CNNTrainer t(cnn);

	t.readTrainFromDir("mnistJPG");
	t.train(1);
	t.testOnTrain();
}
