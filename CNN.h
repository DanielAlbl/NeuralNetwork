#pragma once
#include "Tensor.h"
#include "Classifier.h"

class CNN {
	int N;
	vector<vector<int>> layers; // stores dimensionts of conv and pool layers

	tensor5 W; // weights
	tensor2 B; // biases

	tensor4 V; // temp storage of tensors 

	tensor5 dW; // derivative of weights
	tensor2 dB; // derivative of biases

	Classifier& C; // fully connected network at the end

public:
	void forward(tensor3& x);
	void backward(tensor1& y);
};