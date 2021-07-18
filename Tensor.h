#pragma once
#include "Matrix.h"
#include <cfloat>
using namespace std;

typedef vector<double> tensor1;
typedef vector<tensor1> tensor2;
typedef vector<tensor2> tensor3;
typedef vector<tensor3> tensor4;
typedef vector<tensor4> tensor5;

// only works for odd dim kernels
void CNV(tensor3& prod, tensor3& img, tensor4& kern) {
	int M = img.size(), N = img[0].size(), D = img[0][0].size();
	int d = kern.size(), m = kern[0].size(), n = kern[0][0].size();

	for (int i = 0; i < M; i++) {
		int m_ = min(i + m/2, M), sti = i - m/2;
		for (int j = 0; j < N; j++) {
			int n_ = min(j + n/2, N), stj = j - n/2;
			for (int k_ = 0; k_ < d; k_++) {
				double sum = 0;
				for (int i_ = max(0, sti); i_ < m_; i_++)
					for (int j_ = max(0, stj); j_ < n_; j_++)
						for (int k = 0; k < D; k++)
							sum += img[i_][j_][k] * kern[k_][i_ - sti][j_ - stj][k];
				prod[i][j][k_] = sum;
			}
		}
	}
}

// add convolution derivatives
void CVD(tensor4& dw, tensor3& v, tensor3& dv) {
	int M = v.size(), N = v[0].size(), D = v[0][0].size();
	int d = dw.size(), m = dw[0].size(), n = dw[0][0].size();

	for (int i = 0; i < M; i++) {
		int m_ = min(i + m/2, M), sti = i - m/2;
		for (int j = 0; j < N; j++) {
			int n_ = min(j + n/2, N), stj = j - n/2;
			for (int k_ = 0; k_ < d; k_++) 
				for (int i_ = max(0, sti); i_ < m_; i_++)
					for (int j_ = max(0, stj); j_ < n_; j_++)
						for (int k = 0; k < D; k++)
							dw[k_][i_ - sti][j_ - stj][k] += v[i_][j_][k] * dv[i][j][k_];
		}
	}
}

// Max Pooling
void MPL(tensor3& pool, tensor3& img, int m, int n) {
	int M = img.size(), N = img[0].size(), D = img[0][0].size();
	
	for (int i = 0; i < M; i += m) {
		int m_ = min(i + m, M);
		for (int j = 0; j < N; j += n) {
			int n_ = min(j + n, N);
			for (int k = 0; k < D; k++) {
				double mx = -DBL_MAX;
				for (int i_ = i; i_ < m_; i_++)
					for (int j_ = j; j_ < n_; j_++)
						mx = max(mx, img[i_][j_][k]);
				pool[i/m][j/n][k] = mx;
			}
		}
	}
}

// multiply by derivative of max pooling
void MPD(tensor3& dv, tensor3& v1, tensor3& v2, int m, int n) {
	int M = v1.size(), N = v1[0].size(), D = v1[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				dv[i][j][k] *= v1[i][j][k] == v2[i/m][j/n][k]; 
}

void ADD(tensor3& sum, tensor3& left, tensor1& right) {
	int M = left.size(), N = left[0].size(), D = left[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				sum[i][j][k] = left[i][j][k] + right[k];
}

// just assume activation is relu
void ACT(tensor3& y, tensor3& x) {
	int M = x.size(), N = x[0].size(), D = x[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				y[i][j][k] = max(0.0, x[i][j][k]);
}

// multiply by derivative of relu
void ATD(tensor3& dv, tensor3& v) {
	int M = v.size(), N = v[0].size(), D = v[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				dv[i][j][k] *= v[i][j][k] == 0.0;
}

Matrix flatten(tensor3& t) {
	int M = t.size(), N = t[0].size(), D = t[0][0].size();
	tensor1 v; v.reserve(M * N * D);

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				v.push_back(t[i][j][k]);

	return Matrix(v);
}
