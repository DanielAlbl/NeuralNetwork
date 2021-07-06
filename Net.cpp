#include "Net.h" 

Net::Net() {}

Net::Net(const char* file) {
	read(file);

	V.reserve(N);
	A.reserve(N + 1);

	dW.reserve(N);
	dB.reserve(N);

	for(int i = 0; i < N; i++) {
		V.emplace_back(Matrix(W[i].M, 1));
		A.emplace_back(Matrix(W[i].N, 1));

		dW.emplace_back(Matrix(W[i].M, W[i].N));
		dB.emplace_back(Matrix(W[i].M, 1));
	}

	A.emplace_back(Matrix(W[N-1].M, 1));
}

Net::Net(vector<int> const& sizes) {
	N = sizes.size() - 1;

	W.reserve(N);
	B.reserve(N);

	V.reserve(N);
	A.reserve(N + 1);

	dW.reserve(N);
	dB.reserve(N);

	for(int i = 0; i < N; i++) {
		W.emplace_back(Matrix(sizes[i + 1], sizes[i]));
		B.emplace_back(Matrix(sizes[i + 1], 1));

		V.emplace_back(Matrix(sizes[i + 1], 1));
		A.emplace_back(Matrix(sizes[i], 1));

		dW.emplace_back(Matrix(sizes[i + 1], sizes[i]));
		dB.emplace_back(Matrix(sizes[i + 1], 1));
	}

	A.emplace_back(Matrix(sizes[N], 1));

	init();
}

void Net::forward(Matrix& x) {
	A[0] = x;
	for(int i = 0; i < N; i++) {
		Matrix::MLA(V[i], W[i], A[i], B[i]);
		Matrix::ACT(i == N - 1 ? out : act, A[i + 1], V[i]);
	}
}

void Net::backward(Matrix & y) {
	Matrix::SUB(A[N], A[N], y);
	Matrix::MUL(2, A[N]); // begin backprop with 2(A[N] - y)

	for(int i = N - 1; i > -1; i--) {
		Matrix::ACT(i == N - 1 ? outPrime : actPrime, V[i], V[i]); // V[i] temporarily holds dAct(V[i])/dV[i]
		Matrix::HAD(A[i + 1], A[i + 1], V[i]); // A[i+1] holds dCost/dV[i]

		Matrix::ADD(dB[i], dB[i], A[i + 1]); // add dCost/dB[i] to running sum for the current batch
		Matrix::OUT(dW[i], A[i + 1], A[i]); // add dCost/dW[i] to the running sum

		Matrix::DOT(A[i], W[i], A[i + 1]); // A[i] now holds dCost/dA[i] 
	}
}

void Net::gradDec(double alpha) {
	for(int i = 0; i < N; i++) {
		Matrix::MUL(alpha, dW[i]);
		Matrix::SUB(W[i], W[i], dW[i]);

		Matrix::MUL(alpha, dB[i]);
		Matrix::SUB(B[i], B[i], dB[i]);

		dW[i].setZero();
		dB[i].setZero();
	}
}

void Net::printOutput() {
	A[N].print();
}

int Net::getInDim() {
	return W[0].N;
}

int Net::getOutDim() {
	return W[N].M;
}

void Net::read(const char* file) {
	ifstream fin(file);

	int m, n;
	while(fin >> m and fin >> n) {
		W.emplace_back(Matrix(fin, m, n));
		B.emplace_back(Matrix(fin, m, 1));
	} 

	N = W.size();
}

void Net::write(const char* file) {
	ofstream fout(file);

	for(int i = 0; i < N; i++) {
		fout << W[i].M << " " << W[i].N << endl;
		W[i].write(fout); fout << endl;
		B[i].write(fout); fout << endl;
	}
}

void Net::init() {
	unsigned seed = system_clock::now().time_since_epoch().count();
	default_random_engine gen(seed);

	for(int i = 0; i < N; i++) {
		int m = W[i].M, n = W[i].N;
		normal_distribution<double> dist(0.0, sqrt(2.0 / (m + n)));

		for(int j = 0; j < m; j++) 
			for(int k = 0; k < n; k++) 
				W[i](j, k) = dist(gen);
	}
 }

double Net::reLu(double x) {
	return max(x, 0.0);
}

double Net::reLuPrime(double x) {
	return x > 0 ? 1 : 0;
}

double Net::sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

double Net::sigmoidPrime(double x) {
	double s = sigmoid(x);
	return s * (1 - s);
}
