#include "Classifier.h"

void Classifier::forward(Matrix& x) {
	A[0] = x;
	for (int i = 0; i < N-1; i++) {
		Matrix::MLA(V[i], W[i], A[i], B[i]);
		Matrix::ACT(act, A[i + 1], V[i]);
	}
	Matrix::MLA(V[N-1], W[N-1], A[N-1], B[N-1]);
	softMax(A[N], V[N - 1]);
}

void Classifier::backward(Matrix& y) {
	Matrix::SUB(A[N], A[N], y);
	
	for (int i = N - 1; i > -1; i--) {
		if (i != N - 1) {
			Matrix::ACT(actPrime, V[i], V[i]); // V[i] temporarily holds dAct(V[i])/dV[i]
			Matrix::HAD(A[i + 1], A[i + 1], V[i]); // A[i+1] holds dCost/dV[i]
		}

		Matrix::ADD(dB[i], dB[i], A[i + 1]); // add dCost/dB[i] to running sum for the current batch
		Matrix::OTA(dW[i], A[i + 1], A[i], dW[i]); // add dCost/dW[i] to the running sum

		Matrix::DOT(A[i], W[i], A[i + 1]); // A[i] now holds dCost/dA[i] 
	}
}

void Classifier::softMax(Matrix& y, Matrix& x) {
	float sum = 0;
	for (int i = 0; i < x.M; i++)
		sum += exp(x(i, 0));
	for (int i = 0; i < x.M; i++)
		y(i, 0) = exp(x(i, 0)) / sum;
}
