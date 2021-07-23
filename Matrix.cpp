#include "Matrix.h"

Matrix::Matrix(int size) {
    init(size, size);

    for(int i = 0; i < size; i++)
        elements[i][i] = 1.0;
}

Matrix::Matrix(int m, int n) {
    init(m, n);
}

Matrix::Matrix(ifstream& fin, int m, int n) {
    read(fin, m, n);
}

Matrix::Matrix(vector<float> const& v) {
    init(v.size(), 1);

    for(int i = 0; i < M; i++)
        elements[i][0] = v[i];
}

Matrix::Matrix(vector<vector<float>> const& v) {
    init(v.size(), v[0].size());

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++) 
            elements[i][j] = v[i][j];
}

void Matrix::init(int m, int n) {
    M = m, N = n;

    elements.resize(M);

    for(int i = 0; i < M; i++)
        elements[i].resize(N, 0);
}

void Matrix::setZero() {
    for(auto& v : elements) 
        fill(v.begin(), v.end(), 0);
}

void Matrix::applyAct(float(*act)(float)) {
    for(int i = 0; i < M; i++) 
        for(int j = 0; j < N; j++) 
            elements[i][j] = act(elements[i][j]);
}


void Matrix::print() {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) 
            cout << elements[i][j] << " ";
        cout << "\n";
    }
}

void Matrix::read(ifstream& fin, int m, int n) {
    M = m;
    N = n;

    elements.reserve(M);
    for(int i = 0; i < M; i++) {
        elements[i].reserve(N);
        for(int j = 0; j < N; j++)  
            fin >> elements[i][j];
    }
}

void Matrix::write(ofstream& fout) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) 
            fout << elements[i][j] << " ";
        fout << endl;
    }
}

float & Matrix::operator()(int i, int j) {
    return elements[i][j];
}

Matrix & Matrix::operator=(Matrix const& right) {
    M = right.M;
    N = right.N;
    elements = right.elements;

    return *this;
}

bool Matrix::operator==(Matrix const& right) {
    return elements == right.elements;
}

void Matrix::ADD(Matrix & sum, Matrix & left, Matrix & right) {
    for(int i = 0; i < sum.M; i++) 
        for(int j = 0; j < sum.N; j++) 
            sum(i, j) = left(i, j) + right(i, j);
}

void Matrix::SUB(Matrix & dif, Matrix & left, Matrix & right) {
    for(int i = 0; i < dif.M; i++) 
        for(int j = 0; j < dif.N; j++) 
            dif(i, j) = left(i, j) - right(i, j);
}

void Matrix::MUL(Matrix & prod, Matrix & left, Matrix & right) {
    for(int i = 0; i < prod.M; i++) 
        for(int j = 0; j < prod.N; j++) {
            prod(i, j) = 0;
            for(int k = 0; k < left.N; k++) 
                prod(i, j) += left(i, k) * right(k, j);
        }
}

void Matrix::MUL(float s, Matrix & m) {
    for(int i = 0; i < m.M; i++) 
        for(int j = 0; j < m.N; j++) 
            m(i, j) *= s;
}

void Matrix::HAD(Matrix & prod, Matrix & left, Matrix & right) {
    for(int i = 0; i < prod.M; i++) 
        for(int j = 0; j < prod.N; j++) 
            prod(i, j) = left(i, j) * right(i, j);
}

void Matrix::HDA(Matrix & ans, Matrix & left, Matrix & right, Matrix & acc) {
    for(int i = 0; i < ans.M; i++) 
        for(int j = 0; j < ans.N; j++) 
            ans(i, j) = left(i, j) * right(i, j) + acc(i, j);
}

void Matrix::MLA(Matrix & ans, Matrix & left, Matrix & right, Matrix & acc) {
    for(int i = 0; i < ans.M; i++) 
        for(int j = 0; j < ans.N; j++) {
            ans(i, j) = acc(i, j);
            for(int k = 0; k < left.N; k++) 
                ans(i, j) += left(i, k) * right(k, j);
        }
}

void Matrix::ACT(float(*f)(float), Matrix & y, Matrix & x) {
    for(int i = 0; i < y.M; i++) 
        for(int j = 0; j < y.N; j++) 
            y(i, j) = f(x(i, j));
}

void Matrix::DOT(Matrix & prod, Matrix & left, Matrix & right) {
    for(int i = 0; i < prod.M; i++) 
        for(int j = 0; j < prod.N; j++) {
            prod(i, j) = 0;
            for(int k = 0; k < left.M; k++) 
                prod(i, j) += left(k, i) * right(k, j);
        }
}

void Matrix::OUT(Matrix & prod, Matrix & left, Matrix & right) {
    for(int i = 0; i < prod.M; i++) 
        for(int j = 0; j < prod.N; j++) 
            prod(i, j) = left(i, 0) * right(j, 0);
}

void Matrix::OTA(Matrix& prod, Matrix& left, Matrix& right, Matrix& acc) {
    for (int i = 0; i < prod.M; i++)
        for (int j = 0; j < prod.N; j++)
            prod(i, j) = left(i, 0) * right(j, 0) + acc(i, j);
}

Matrix operator*(Matrix& left, Matrix& right) {
    Matrix prod(left.M, right.N);
    Matrix::MUL(prod, left, right);
    return prod;
}

Matrix operator+(Matrix & left, Matrix & right) {
    Matrix sum(left.M, left.N);
    Matrix::ADD(sum, left, right);
    return sum;
}

Matrix operator-(Matrix & left, Matrix & right) {
    Matrix dif(left.M, left.N);
    Matrix::SUB(dif, left, right);
    return dif;
}
