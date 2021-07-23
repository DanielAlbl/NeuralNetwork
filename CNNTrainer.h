#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <climits>
#include <numeric>
#include "CNN.h"
namespace fs = filesystem;

class CNNTrainer {
	CNN& C;

	tensor4 Xtrain, Xtest;
	tensor2 Ytrain, Ytest;
	vector<int> order;
	
	tensor3 mean, std;

	int maxRows = INT_MAX;
	int batchSize = 32;
	float stepSize = 0.1;

	int trainSize = 0;
	int testSize = 0;

	tensor1 to1hot(int x, int size);

	void standardizeTrain();
	void standardizeTest();

public:
	CNNTrainer(CNN& cnn);

	void readTrainFromDir(const char* dir); // reads images organized into classes by directory

	void train(int epochs);
	float testOnTrain();
};
