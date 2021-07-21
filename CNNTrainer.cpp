#include "CNNTrainer.h"

tensor1 CNNTrainer::to1hot(int x, int size) {
	tensor1 t(size, 0);
	t[x] = 1;
	return t;
}

void CNNTrainer::readFromDir(const char* dir) {
	int i = 0; 
	for (auto& subdr : fs::directory_iterator(dir)) {
		int size = distance(fs::directory_iterator(subdr), fs::directory_iterator{});
		tensor1 oneHot = to1hot(i++, size);
		for (auto& file : fs::directory_iterator(subdr)) {
			cv::Mat mat = cv::imread(file.path().string());
			mat.convertTo(mat, CV_64F);

			Xtrain.emplace_back(tensor3((double*) mat.data, (double*) mat.data + mat.rows * mat.cols * mat.channels()));
			Ytrain.push_back(oneHot);
		}
	}
}
