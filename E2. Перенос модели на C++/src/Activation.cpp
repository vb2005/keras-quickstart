#include "Activation.h"

float linear(float data) {
	return data;
}
float step(float data) {
	if (data >= 0) return 1;
	return 0;
}
float elu(float data) {
	if (data > 0) return data;
	return exp(data) - 1;
}
float relu(float data) {
	if (data > 0) return data;
	return 0;
}
float sigmoid(float data) {
	return 1 / (1 + exp(-data));
}
float softSign(float data) {
	return data / (1 + abs(data));
}
float silu(float data) {
	return data * sigmoid(data);
}
float softMax(array1D src, float data) {
	float sum = 0;
	for (int i = 0; i < src.size; i++)
		sum += src.data[i];
	return data / sum;
}

void Activation(array3D& src, array3D& dst, float (*actType)(float)) {
	dst.init(src.size_1, src.size_2, src.size_3);
	for (int i = 0; i < src.size_1; i++)
		for (int j = 0; j < src.size_2; j++)
			for (int k = 0; k < src.size_3; k++)
				dst.data[i][j][k] = actType(src.data[i][j][k]);
}