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

	float(*act)(float) = reLu;    // activation function
	float(*out)(float) = sigmoid; // output function

	float(*actPrime)(float) = reLuPrime; 
	float(*outPrime)(float) = sigmoidPrime;

public:
	Net();
	Net(const char* file);
	Net(vector<int> const& sizes);
	virtual ~Net() {}

	void init();

	virtual void forward(Matrix& x);
	virtual void backward(Matrix& y);

	void gradDec(float alpha);
	void printOutput();

	int getInDim();
	int getOutDim();
	Matrix& getInputGrad();

	void read(const char* file);
	void write(const char* file);

	int getOutputClass();
	Matrix predict(Matrix& x);

	static float reLu(float x);
	static float reLuPrime(float x);

	static float sigmoid(float x);
	static float sigmoidPrime(float x);
};

