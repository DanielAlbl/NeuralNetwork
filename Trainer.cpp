#include "Trainer.h"

void Trainer::initOrder() {
	if (order.size() == (uint)dataPoints) return;
	order.resize(dataPoints);
	iota(order.begin(), order.end(), 0);
}

Trainer::Trainer(Net& n)
	: N(n), inDim(N.getInDim()), outDim(N.getOutDim()) {}

void Trainer::readX(const char * file) {
	ifstream fin(file, ios::in);

	string line, num;
	vector<double> tmp(inDim);

	int i = 0;
	for(; i < maxRows and getline(fin, line); i++) {
		stringstream ss(line);
		for(int j = 0; getline(ss, num, ','); j++) 
			tmp[j] = stod(num);
		X.emplace_back(Matrix(tmp));
	}

	dataPoints = i;

	standardize();
	initOrder();
}

void Trainer::readY(const char * file) {
	ifstream fin(file, ios::in);

	string line, num;
	vector<double> tmp(outDim);

	int i = 0;
	for(; i < maxRows and getline(fin, line); i++) {
		stringstream ss(line);
		for(int j = 0; getline(ss, num, ','); j++) 
			tmp[j] = stod(num);
		Y.emplace_back(Matrix(tmp));
	}

	dataPoints = i;
	initOrder();
}

void Trainer::train(int epochs) {
	for(int i = 0; i < epochs; i++) {
		shuffle(order.begin(), order.end(), mt19937());
		int j = 0;
		while (j < dataPoints) {
			int bound = min(j + batchSize, dataPoints), size = bound - j;
			for (; j < bound; j++) {
				N.forward(X[order[j]]);
				N.backward(Y[order[j]]);
			}
			N.gradDec(stepSize / size);
		}
	}
}

double Trainer::test() {
	int cnt = 0;
	for(int i = 0; i < dataPoints; i++) 
		cnt += N.predict(X[i]) == Y[i];
			
	return (double)cnt / (double)X.size();
}

void Trainer::standardize() {
	for (int j = 0; j < X[0].M; j++) {
		mean = 0, std = 0;
		
		for (int i = 0; i < dataPoints; i++) 
			mean += X[i](j, 0);
		mean /= dataPoints;

		for (int i = 0; i < dataPoints; i++)
			X[i](j, 0) -= mean, std += X[i](j, 0) * X[i](j, 0);
		std = sqrt(std / dataPoints);
		
		for (int i = 0; i < dataPoints; i++)
			X[i](j, 0) /= std;
	}
}
