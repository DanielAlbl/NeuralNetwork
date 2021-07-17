#include "CNN.h"

void CNN::forward(tensor3& x) {
	int j = 0;
	V[0] = x;

	for (int i = 0; i < layers.size(); i++) {
		if (layers[i].size() == 3) {
			CNV(V[i + 1], V[i], W[j]);
			ADD(V[i + 1], V[i + 1], B[j]);
			ACT(V[i + 1], V[i + 1]);
			j++;
		}
		else
			MPL(V[i + 1], V[i], layers[i][0], layers[i][1]);
	}
}
