#include "Dense.h"

void Dense(array1D& src, array1D& dst, array2D& weights, int size) {
	dst.init(size);
	dst.set(0);
	for (int i = 0; i < src.size; i++)
		for (int j = 0; j < size; j++)
			dst.data[size] += src.data[i] * weights.data[j][i];
}

void Dense(array1D& src, array1D& dst, array2D& weights, array1D& bias, int size) {
	dst.init(size);
	dst.set(0);
	for (int i = 0; i < size; i++)
		dst.data[i] = bias.data[i];

	for (int i = 0; i < src.size; i++)
		for (int j = 0; j < size; j++)
			dst.data[j] += src.data[i] * weights.data[j][i];
}