#include "Trainer.h"

void Trainer::initOrder() {
	if (order.size() == dataPoints) return;
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
	for(int i = 0; i < X.size(); i++) 
		cnt += N.predict(X[i]) == Y[i];
			
	return (double)cnt / (double)X.size();
}
