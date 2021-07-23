#include "CNNTrainer.h"

tensor1 CNNTrainer::to1hot(int x, int size) {
	tensor1 t(size, 0.0);
	t[x] = 1.0;
	return t;
}

CNNTrainer::CNNTrainer(CNN& cnn) : C(cnn) {}

void CNNTrainer::standardizeTrain() {
	cout << "Reading Data...\n";

	int M = Xtrain[0].size(), N = Xtrain[0][0].size(), D = Xtrain[0][0][0].size();
	mean = makeTensor3(M, N, D);
	std = makeTensor3(M, N, D);

	for (int j = 0; j < M; j++) 
		for(int k = 0; k < N; k++) 
			for (int l = 0; l < D; l++) {
				for (int i = 0; i < trainSize; i++) 
					mean[j][k][l] += Xtrain[i][j][k][l];

				mean[j][k][l] /= trainSize;

				for (int i = 0; i < trainSize; i++)
					Xtrain[i][j][k][l] -= mean[j][k][l], std[j][k][l] += Xtrain[i][j][k][l] * Xtrain[i][j][k][l];
				std[j][k][l] = sqrtf(std[j][k][l] / trainSize);

				for (int i = 0; i < trainSize; i++)
					Xtrain[i][j][k][l] = std[j][k][l] ? Xtrain[i][j][k][l] / std[j][k][l] : 0.0;
			}

	if (testSize)
		standardizeTest();
}

void CNNTrainer::standardizeTest() {
	if (trainSize == 0) return;

	int M = Xtrain[0].size(), N = Xtrain[0][0].size(), D = Xtrain[0][0][0].size();
	for (int i = 0; i < testSize; i++)
		for (int j = 0; j < M; j++)
			for (int k = 0; k < N; k++)
				for (int l = 0; l < D; l++) 
					Xtest[i][j][k][l] = std[j][k][l] ? (Xtest[i][j][k][l] - mean[j][k][l]) / std[j][k][l] : 0.0;
}

void CNNTrainer::readTrainFromDir(const char* dir) {
	int size = distance(fs::directory_iterator(dir), fs::directory_iterator{});
	int i = 0, cls = 0;
	for (auto& subdr : fs::directory_iterator(dir)) {
		tensor1 oneHot = to1hot(cls++, size);
		int cnt = 0;
		for (auto& file : fs::directory_iterator(subdr.path())) {
			if(cnt++ >= maxRows/size) break;

			cv::Mat mat = cv::imread(file.path().string());
			cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
			mat.convertTo(mat, CV_32F);

			Xtrain.emplace_back(makeTensor3(mat.rows, mat.cols, mat.channels()));

			for (int j = 0; j < mat.rows; j++)
				for (int k = 0; k < mat.cols; k++)
					Xtrain[i][j][k][0] = mat.at<float>(j, k);

			Ytrain.push_back(oneHot);
			i++;
		}
	}

	order.resize(trainSize = Xtrain.size());
	iota(order.begin(), order.end(), 0);

	standardizeTrain();
}

void CNNTrainer::train(int epochs) {
	for (int i = 0; i < epochs; i++) {
		shuffle(order.begin(), order.end(), mt19937());
		int j = 0;
		while (j < trainSize) {
			int bound = min(j + batchSize, trainSize), size = bound - j;
			for (; j < bound; j++) {
				C.forward(Xtrain[order[j]]);
				C.backward(Ytrain[order[j]]);
			}
			C.gradDec(stepSize / size);
			cout << "\rTraining: " << 100.0 * (i * trainSize + j) / (epochs * trainSize) << "%            ";
			cout.flush();
		}
	}
	cout << endl;
}

float CNNTrainer::testOnTrain() {
	int cnt = 0;
	for (int i = 0; i < trainSize; i++) {
		cnt += C.predict(Xtrain[i]) == Ytrain[i];
		cout << "\rTesting: " << 100.0 * (i + 1) / trainSize << "%          ";
		cout.flush();
	}

	float acc = (float) cnt / (float) trainSize;
	cout << "\nAccuracy: " << acc << '\n';
	return acc;
}
