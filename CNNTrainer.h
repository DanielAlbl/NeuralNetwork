#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "CNN.h"
namespace fs = filesystem;

class CNNTrainer {
	CNN& cnn;

	tensor4 Xtrain, Xtest;
	tensor2 Ytrain, Ytest;
	
	tensor3 mean, std;

	vector<int> order;

	int maxRows = INT_MAX;
	int batchSize = 32;
	double stepSize = 0.1;

	tensor1 to1hot(int x, int size);

	void readFromDir(const char* dir); // reads images organized into classes by directory
public:
};