#include "CNN.h"

CNN::CNN(vector<int> const& inDim, vector<int> const& conv, vector<int> const& fc) : layers(conv) {
	int m = inDim[0], n = inDim[1], d = inDim[2];
	N = layers.size(), M = N-count(layers.begin(), layers.end(), 0);

	W.resize(M); dW.resize(M);
	B.resize(M); dB.resize(M);
	V.resize(N + 1); dV.resize(N + 1);

	V[0] = dV[0] = makeTensor3(m, n, d);

	int j = 0;
	for (int i = 0; i < N; i++) {
		if (layers[i]) {
			W[j] = dW[j] = makeTensor4(layers[i], km, kn, V[i][0][0].size());
			B[j] = dB[j] = makeTensor1(layers[i]);
			V[i + 1] = dV[i + 1] = makeTensor3(m, n, layers[i]);
			j++;
		}
		else {
			m = (m + pm - 1) / pm, n = (n + pn - 1) / pn;
			V[i + 1] = dV[i + 1] = makeTensor3(m, n, V[i][0][0].size());
		}
	}

	init();

	// add input layer to fully connected net of size of flattend output of conv net
	vector<int> tmp{ (int) (V[N].size() * V[N][0].size() * V[N][0].size()) };
	tmp.insert(tmp.end(), fc.begin(), fc.end());
	C = move(Classifier(tmp));
}

void CNN::init() {
	int kDim = km * kn;
	random_device gen;
	for (auto& i : W) {
		normal_distribution<double> dist(0, sqrt(2.0 / (kDim * (double)i[0][0].size())));
		for (auto& j : i)
			for (auto& k : j)
				for (auto& l : k)
					for(auto& m : l)
						m = dist(gen);
	}
}

void CNN::forward(tensor3& x) {
	int j = 0;
	V[0] = x;

	for (int i = 0; i < N; i++) {
		if (layers[i]) {
			setZero(V[i + 1]);
			CNV(V[i + 1], V[i], W[j], abc); // convolve
			ADD(V[i + 1], V[i + 1], B[j]); // apply biases
			ACT(V[i + 1], V[i + 1]); // apply activation
			j++;
		}
		else
			MPL(V[i + 1], V[i], pm, pn); // max pooling
	}

	// feed foward through fully connected layers
	Matrix m(V[N].size() * V[N][0].size() * V[N][0][0].size(), 0);
	flatten(m, V[N]);
	C.forward(m);
}

void CNN::backward(tensor1& y) {
	// backpropagate fully connected layers
	Matrix m1(y);
	C.backward(m1);
	Matrix& m2 = C.getInputGrad();
	unFlatten(dV[N], m2);
	int j = M-1;
	
	for (int i = N - 1; i > -1; i--) {
		if (layers[i]) {
			ATD(dV[i + 1], dV[i + 1]); // multiply by activation derivative
			ADD(dB[j], dB[j], dV[i + 1]); // add bias derivatives to running batch sum
			CNV(dV[i + 1], V[i], dW[j], cab); // add weight derivatives to running batch sum
			setZero(dV[i]);
			CNV(V[i + 1], dV[i], dW[j], bac); // calculate derivative w/ respect to V[i] aka dV[i]
			j--;
		}
		else
			MPD(dV[i], V[i], V[i + 1], pm, pn); // calc dV[i] for in the case of max pooling
	}
}

void CNN::gradDec(double alpha) {
	MUL(W, alpha, W);
	MUL(B, alpha, B);

	SUB(W, W, dW);
	SUB(B, B, dB);
	
	setZero(dW);
	setZero(dB);
}