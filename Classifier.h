#pragma once
#include "Net.h"

class Classifier : public Net {
public:
	using Net::Net;

	void forward(Matrix& x);
	void backward(Matrix& y);

	void softMax(Matrix& y, Matrix& x);
};

