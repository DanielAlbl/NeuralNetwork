#include "Trainer.h"

void Trainer::initOrder() {
	order.resize(trainSize);
	iota(order.begin(), order.end(), 0);
}

Trainer::Trainer(Net& n)
	: N(n), inDim(N.getInDim()), outDim(N.getOutDim()) {}

void Trainer::read(vector<Matrix>& v, const char * file, int size) {
	ifstream fin(file, ios::in);
	string line, num;
	vector<double> tmp(size);

	for (int i = 0; i < maxRows and getline(fin, line); i++) {
		stringstream ss(line);
		for (int j = 0; getline(ss, num, ','); j++) 
			tmp[j] = stod(num);
		v.emplace_back(Matrix(tmp));
	}
}

void Trainer::readTraining(const char * x, const char * y) {
	read(Xtrain, x, inDim);
	read(Ytrain, y, outDim);

	trainSize = Xtrain.size();

	standardizeTrain();
	initOrder();
}

void Trainer::readTesting(const char * x, const char * y) {
	read(Xtest, x, inDim);
	read(Ytest, y, outDim);

	testSize = Xtest.size();

	standardizeTest();
}

void Trainer::train(int epochs) {
	for (int i = 0; i < epochs; i++) {
		shuffle(order.begin(), order.end(), mt19937());
		int j = 0;
		while (j < trainSize) {
			int bound = min(j + batchSize, trainSize), size = bound - j;
			for (; j < bound; j++) {
				N.forward(Xtrain[order[j]]);
				N.backward(Ytrain[order[j]]);
			}
			N.gradDec(stepSize / size);
		}
	}
}

double Trainer::test() {
	int cnt = 0;
	for (int i = 0; i < testSize; i++) 
		cnt += N.predict(Xtest[i]) == Ytest[i];
			
	return (double)cnt / (double)testSize;
}

void Trainer::standardizeTrain() {
	mean.resize(inDim, 0); 
	std.resize(inDim, 0);

	for (int j = 0; j < Xtrain[0].M; j++) {
		for (int i = 0; i < trainSize; i++) 
			mean[j] += Xtrain[i](j, 0);
		mean[j] /= trainSize;

		for (int i = 0; i < trainSize; i++)
			Xtrain[i](j, 0) -= mean[j], std[j] += Xtrain[i](j, 0) * Xtrain[i](j, 0);
		std[j] = sqrt(std[j] / trainSize);
		
		for (int i = 0; i < trainSize; i++)
			Xtrain[i](j, 0) /= std[j];
	}

	if(testSize)
		standardizeTest();
}

void Trainer::standardizeTest() {
	if(trainSize == 0) return;
	for (int i = 0; i < testSize; i++)
		for (int j = 0; j < inDim; j++)
			Xtest[i](j, 0) = (Xtest[i](j, 0) - mean[j]) / std[j];
}

