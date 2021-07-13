#pragma once
#include "Net.h"
#include "Classifier.h"
#include <climits>
#include <numeric>

class Trainer {
	Net& N;

	vector<Matrix> X;
	vector<Matrix> Y;

	vector<int> order;

	int dataPoints = 0;
	int batchSize = 150;
	int maxRows = INT_MAX;

	int inDim;
	int outDim;

	double stepSize = 0.1;

	double mean;
	double std;

	void initOrder();

public:
	Trainer(Net& n);

	void readX(const char* file);
	void readY(const char* file);

	void train(int itr);
	double test();

	void standardize();
};

