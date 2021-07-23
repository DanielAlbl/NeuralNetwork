#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

class Matrix {
    vector<vector<float>> elements;

public:
    int M, N;

    Matrix(int size);
    Matrix(int m, int n);
    Matrix(ifstream& fin, int m, int n);
    Matrix(vector<float> const& v);
    Matrix(vector<vector<float>> const& v);
    ~Matrix() {}

    void init(int m, int n);
    void setZero();
    void applyAct(float(*act)(float));
    void print();

    void read(ifstream& fin, int m, int n);
    void write(ofstream& fout);

    float& operator()(int i, int j);
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
    static void MUL(float s, Matrix& m); // Scalar Multiply
    static void MLA(Matrix& ans, Matrix& left, Matrix& right, Matrix& acc); // Multiply Accumulate

    static void HAD(Matrix& prod, Matrix& left, Matrix& right); // Hadamard Product
    static void HDA(Matrix& ans, Matrix& left, Matrix& right, Matrix& acc); // Hadamard Accumulate

    static void DOT(Matrix& prod, Matrix& left, Matrix& right); // Dot Product
    static void OUT(Matrix& prod, Matrix& left, Matrix& right); // Outer Product
    static void OTA(Matrix& prod, Matrix& left, Matrix& right, Matrix& acc); // Outer Product Accumulate

    static void ACT(float(*f)(float), Matrix& y, Matrix& x); // Action aka perform function elementwise
};

