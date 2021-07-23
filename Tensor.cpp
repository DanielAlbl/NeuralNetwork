#include "Tensor.h"

// Convolution helpers
void abc(float& a, float& b, float& c) { a += b * c; }
void bac(float& a, float& b, float& c) { b += a * c; }
void cab(float& a, float& b, float& c) { c += a * b; }

void setZero(tensor2& t) {
	for (auto& i : t)
		fill(i.begin(), i.end(), 0.0);
}

void setZero(tensor3& t) { for (auto& i : t) setZero(i); }

void setZero(tensor4& t) { for (auto& i : t) setZero(i); }

void setZero(tensor5& t) { for (auto& i : t) setZero(i); }

// Convolution
// only works for odd dim kernels
void CNV(tensor3& prod, tensor3& img, tensor4& kern, void(*f)(float&, float&, float&)) {
	int M = img.size(), N = img[0].size(), D = img[0][0].size();
	int d = kern.size(), m = kern[0].size(), n = kern[0][0].size();

	for (int i = 0; i < M; i++) {
		int m_ = min(i + m / 2, M), sti = i - m / 2;
		for (int j = 0; j < N; j++) {
			int n_ = min(j + n / 2, N), stj = j - n / 2;
			for (int k_ = 0; k_ < d; k_++) 
				for (int i_ = max(0, sti); i_ < m_; i_++) 
					for (int j_ = max(0, stj); j_ < n_; j_++) 
						for (int k = 0; k < D; k++) 
							f(prod[i][j][k_], img[i_][j_][k], kern[k_][i_ - sti][j_ - stj][k]);
		}
	}
}

// Max Pooling
void MPL(tensor3& pool, tensor3& img, int m, int n) {
	int M = img.size(), N = img[0].size(), D = img[0][0].size();

	for (int i = 0; i < M; i += m) {
		int m_ = fminf(i + m, M);
		for (int j = 0; j < N; j += n) {
			int n_ = fminf(j + n, N);
			for (int k = 0; k < D; k++) {
				float mx = -DBL_MAX;
				for (int i_ = i; i_ < m_; i_++)
					for (int j_ = j; j_ < n_; j_++)
						mx = fmaxf(mx, img[i_][j_][k]);
				pool[i / m][j / n][k] = mx;
			}
		}
	}
}

// multiply by derivative of max pooling
void MPD(tensor3& dv1, tensor3& dv2, tensor3& v1, tensor3& v2) {
	int M = v1.size(), N = v1[0].size(), D = v1[0][0].size();
	int m = (M + v2.size() - 1) / v2.size(), n = (N + v2[0].size() - 1) / v2[0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				dv1[i][j][k] = (v1[i][j][k] == v2[i/m][j/n][k]) * dv2[i/m][j/n][k];
}

void ADD(tensor3& sum, tensor3& left, tensor1& right) {
	int M = left.size(), N = left[0].size(), D = left[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				sum[i][j][k] = left[i][j][k] + right[k];
}

void ADD(tensor1& sum, tensor1& left, tensor3& right) {
	int M = right.size(), N = right[0].size(), D = right[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				sum[k] = left[k] + right[i][j][k];
}

void SUB(tensor5& diff, tensor5& left, tensor5& right) {
	int M = left.size(), N = left[0].size(), D = left[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				SUB(diff[i][j][k], left[i][j][k], right[i][j][k]);
}

void SUB(tensor2& diff, tensor2& left, tensor2& right) {
	int M = left.size(), N = left[0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			diff[i][j] = left[i][j] - right[i][j];
}

void MUL(tensor5& prod, float s, tensor5& t) {
	int M = t.size(), N = t[0].size(), D = t[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				MUL(prod[i][j][k], s, t[i][j][k]);
}

void MUL(tensor2& prod, float s, tensor2& t) {
	int M = t.size(), N = t[0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			prod[i][j] = s * t[i][j];
}

// just assume activation is relu
void ACT(tensor3& y, tensor3& x) {
	int M = x.size(), N = x[0].size(), D = x[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				y[i][j][k] = fmaxf(0.0, x[i][j][k]);
}

// multiply by derivative of relu
void ATD(tensor3& dv, tensor3& v) {
	int M = v.size(), N = v[0].size(), D = v[0][0].size();
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				dv[i][j][k] *= v[i][j][k] != 0.0;
}

void flatten(Matrix& m, tensor3& t) {
	int M = t.size(), N = t[0].size(), D = t[0][0].size(), i_ = 0;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++) 
				m(i_++, 0) = t[i][j][k];
}

void unFlatten(tensor3& t, Matrix& m) {
	int M = t.size(), N = t[0].size(), D = t[0][0].size(), i_ = 0;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < D; k++)
				t[i][j][k] = m(i_++, 0.0);
}

tensor1 makeTensor1(int m) {
	return tensor1(m, 0.0);
}

tensor2 makeTensor2(int m, int n) {
	return tensor2(m, makeTensor1(n));
}

tensor3 makeTensor3(int m, int n, int d) {
	return tensor3(m, makeTensor2(n, d));
}

tensor4 makeTensor4(int m, int n, int d, int e) {
	return tensor4(m, makeTensor3(n, d, e));
}
