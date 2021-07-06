#include "Trainer.h"

int main() {
	vector<int> sizes = { 200,400,400,200 };
	vector<double> v(200,1);

	Net n(sizes);
	Matrix m(v);

	n.forward(m);
	n.backward(m);
	 
	return 0;
}
