#pragma once
#include "Matrix.h"
#include <cfloat>
using namespace std;

typedef vector<double> tensor1;
typedef vector<tensor1> tensor2;
typedef vector<tensor2> tensor3;
typedef vector<tensor3> tensor4;
typedef vector<tensor4> tensor5;

// Convolution helpers
void abc(double& a, double& b, double& c);
void bac(double& a, double& b, double& c);
void cab(double& a, double& b, double& c);

void setZero(tensor3& t);

void CNV(tensor3& prod, tensor3& img, tensor4& kern, void(*f)(double&, double&, double&));
void MPL(tensor3& pool, tensor3& img, int m, int n);
void MPD(tensor3& dv, tensor3& v1, tensor3& v2, int m, int n);
void ADD(tensor3& sum, tensor3& left, tensor1& right);
void ADD(tensor1& sum, tensor1& left, tensor3& right);
void ACT(tensor3& y, tensor3& x);
void ATD(tensor3& dv, tensor3& v);

void flatten(Matrix& m, tensor3& t);
void unFlatten(tensor3& t, Matrix& m);

tensor1 makeTensor1(int m);
tensor2 makeTensor2(int m, int n);
tensor3 makeTensor3(int m, int n, int d);
tensor4 makeTensor4(int m, int n, int d, int e);