#pragma once
#include "Net.h"

class Trainer {
	Net* N;

	vector<Matrix> X;
	vector<Matrix> Y;

	int dataPoints = 0;
	int batchSize = 32;

	int inDim;
	int outDim;

	double stepSize = 0.0001;

public:
	Trainer(Net& n);

	void readX(const char* file);
	void readY(const char* file);

	void train(int itr);
};
