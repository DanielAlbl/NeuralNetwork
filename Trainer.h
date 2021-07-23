#pragma once
#include "Net.h"
#include "Classifier.h"
#include <climits>
#include <numeric>

class Trainer {
	Net& N;

	vector<Matrix> Xtrain, Ytrain, Xtest, Ytest;
	vector<int> order;
	vector<float> mean, std;

	int trainSize = 0;
	int testSize = 0;
	int batchSize = 32;
	int maxRows = INT_MAX;

	int inDim, outDim;

	float stepSize = 0.1;

	void initOrder();
	void read(vector<Matrix>& v, const char * file, int size);

	void standardizeTrain();
	void standardizeTest();

public:
	Trainer(Net& n);

	void readTraining(const char* x, const char* y);
	void readTesting(const char* x, const char* y);

	void train(int itr);
	float test();
};

