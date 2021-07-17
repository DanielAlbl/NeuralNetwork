#pragma once
#include <vector>
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

	for (int k_ = 0; k_ < d; k_++) 
		for (int i = 0; i < M; i++) {
			int m_ = min(i + m/2, M), sti = i - m/2;
			for (int j = 0; j < N; j++) {
				int n_ = min(j + n/2, N), stj = j - n/2;
				double sum = 0;
				for (int i_ = max(0, sti); i_ < m_; i_++)
					for(int j_ = max(0, stj); j_ < n_; j_++)
						for (int k = 0; k < D; k++) 
							sum += img[i_][j_][k] * kern[k_][i_-sti][j_-stj][k];
				prod[i][j][k_] = sum;
			}
		}
}

void MPL(tensor3& pool, tensor3& img, int m, int n) {
	int M = img.size(), N = img[0].size(), D = img[0][0].size();

	for(int k = 0; k < D; k++) 
		for (int i = 0; i < M; i += m) {
			int m_ = min(i + m, M);
			for (int j = 0; j < N; j += n) {
				int n_ = min(j + n, N);
				double mx = -DBL_MAX;
				for (int i_ = i; i_ < m_; i_++)
					for (int j_ = j; j_ < n_; j_++)
						mx = max(mx, img[i_][j_][k]);
				pool[i/m][j/n][k] = mx;
			}
		}
}

void ADD(tensor3& diff, tensor3& left, tensor1& right) {
	int M = left.size(), N = left[0].size(), D = left[0][0].size();

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				diff[i][j][k] = left[i][j][k] + right[k];
}

// just assume activation is relu
void ACT(tensor3& y, tensor3& x) {
	int M = x.size(), N = x[0].size(), D = x[0][0].size();

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				y[i][j][k] = max(0.0, x[i][j][k]);
}
