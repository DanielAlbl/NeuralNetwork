#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

class Matrix {
    vector<vector<double>> elements;

public:
    int M, N;

    Matrix(int size);
    Matrix(int m, int n);
    Matrix(ifstream& fin, int m, int n);
    Matrix(vector<double> const& v);
    Matrix(vector<vector<double>> const& v);
    ~Matrix() {}

    void init(int m, int n);
    void setZero();
    void applyAct(double(*act)(double));
    void print();

    void read(ifstream& fin, int m, int n);
    void write(ofstream& fout);

    double& operator()(int i, int j);
    Matrix& operator=(Matrix const& right);
    bool operator==(Matrix const& right);

    friend Matrix operator*(Matrix& left, Matrix& right);
    friend Matrix operator+(Matrix& left, Matrix& right);
    friend Matrix operator-(Matrix& left, Matrix& right);

    // can specify the ans matrix to avoid extra copying
    // also reminds me of Assembly
    static void ADD(Matrix& sum, Matrix& left, Matrix& right);
    static void SUB(Matrix& dif, Matrix& left, Matrix& right);

    static void MUL(Matrix& prod, Matrix& left, Matrix& right); // Matrix Multiply
    static void MUL(double s, Matrix& m); // Scalar Multiply
    static void MLA(Matrix& ans, Matrix& left, Matrix& right, Matrix& acc); // Multiply Accumulate

    static void HAD(Matrix& prod, Matrix& left, Matrix& right); // Hadamard Product
    static void HDA(Matrix& ans, Matrix& left, Matrix& right, Matrix& acc); // Hadamard Accumulate

    static void DOT(Matrix& prod, Matrix& left, Matrix& right); // Dot Product
    static void OUT(Matrix& prod, Matrix& left, Matrix& right); // Outer Product
    static void OTA(Matrix& prod, Matrix& left, Matrix& right, Matrix& acc); // Outer Product Accumulate

    static void ACT(double(*f)(double), Matrix& y, Matrix& x); // Action aka perform function elementwise
    static void CNV(Matrix& conv, Matrix& img, Matrix& kernel); // Convolution with padding of 0 pixels
};

