#pragma once
#include "Net.h"
#include "classifier.h"
#include <climits>

class Trainer {
	Net& N;

	vector<Matrix> X;
	vector<Matrix> Y;

	int dataPoints = 0;
	int batchSize = 32;
	int maxRows = INT_MAX;

	int inDim;
	int outDim;

	double stepSize = 0.0001;

public:
	Trainer(Net& n);

	void readX(const char* file);
	void readY(const char* file);

	void train(int itr);
	double test();
};

