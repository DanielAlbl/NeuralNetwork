#include "Trainer.h"

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
}

void Trainer::train(int itr) {
	srand(time(NULL));
	int rnd;

	for(int i = 0; i < itr; i++) {
		for(int j = 0; j < batchSize; j++) {
			rnd = rand() % dataPoints;
			N.forward(X[rnd]);
			N.backward(Y[rnd]);
		}
		N.gradDec(stepSize / batchSize);
	}
}

double Trainer::test() {
	int cnt = 0;
	for(int i = 0; i < X.size(); i++) 
		cnt += N.predict(X[i]) == Y[i];
			
	return (double)cnt / (double)X.size();
}
