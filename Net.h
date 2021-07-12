#pragma once
#include "Matrix.h"
#include <cfloat>
#include <algorithm>
#include <random>
#include <chrono>
using namespace chrono;

class Net {
protected:
	vector<Matrix> W; // Weights
	vector<Matrix> B; // Biases

	vector<Matrix> V; // Vectors pre-activation
	vector<Matrix> A; // Vectors post-activation

	vector<Matrix> dW; // change in weights
	vector<Matrix> dB; // change in biases

	int N = 0; // Number of Matrices (layers - 1)

	double(*act)(double) = reLu;    // activation function
	double(*out)(double) = sigmoid; // output function

	double(*actPrime)(double) = reLuPrime; 
	double(*outPrime)(double) = sigmoidPrime;

public:
	Net();
	Net(const char* file);
	Net(vector<int> const& sizes);
	virtual ~Net() {}

	void init();

	virtual void forward(Matrix& x);
	virtual void backward(Matrix& y);

	void gradDec(double alpha);
	void printOutput();

	int getInDim();
	int getOutDim();

	void read(const char* file);
	void write(const char* file);

	Matrix predict(Matrix& x);

	static double reLu(double x);
	static double reLuPrime(double x);

	static double sigmoid(double x);
	static double sigmoidPrime(double x);
};

