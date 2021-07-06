#pragma once
#include "Matrix.h"
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

using namespace chrono;

class Net {
	vector<Matrix> W; // Weights
	vector<Matrix> B; // Biases

	vector<Matrix> V; // Vectors pre-activation
	vector<Matrix> A; // Vectors post-activation

	vector<Matrix> dW; // change in weights
	vector<Matrix> dB; // change in biases

	int N = 0; // Number of Matrices (layers - 1)

	double(*act)(double) = reLu; // activation function
	double(*out)(double) = reLu; // output function

	double(*actPrime)(double) = reLuPrime; 
	double(*outPrime)(double) = reLuPrime;

public:
	Net();
	Net(const char* file);
	Net(vector<int> const& sizes);
	~Net() {}

	void init();

	void forward(Matrix& x);
	void backward(Matrix& y);

	void gradDec(double alpha);
	void printOutput();

	int getInDim();
	int getOutDim();

	void read(const char* file);
	void write(const char* file);

	static double reLu(double x);
	static double reLuPrime(double x);

	static double sigmoid(double x);
	static double sigmoidPrime(double x);
};

