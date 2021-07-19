#pragma once
#include <random>
#include "Tensor.h"
#include "Classifier.h"
using namespace std;

class CNN {
	int N, M;
	vector<int> layers; // type of layer and number of filters

	tensor5 W; // weights
	tensor2 B; // biases
	tensor4 V; // temp storage of tensors 

	tensor5 dW; // derivative of weights
	tensor2 dB; // derivative of biases
	tensor4 dV; // derivative of temp tensors

	Classifier C; // fully connected network at the end

	int km = 3, kn = 3; // kernel dimensions
	int pm = 2, pn = 2; // pooling dimensions

public:
	CNN(vector<int> const& inDim, vector<int> const& conv, vector<int> const& fc);

	void init();

	void forward(tensor3& x);
	void backward(tensor1& y);
};