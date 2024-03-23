#include "Flatten.h"

void Flatten(array4D& src, array1D& dst) {
	dst.init(src.size_1 * src.size_2 * src.size_3 * src.size_4);
	int x = 0;
	for (int i = 0; i < src.size_1; i++)
		for (int j = 0; j < src.size_2; j++)
			for (int k = 0; k < src.size_3; k++) {
				for (int l = 0; l < src.size_4; k++) {
					dst.data[x] = src.data[i][j][k][l];
					x++;
				}
			}
}

void Flatten(array3D& src, array1D& dst) {
	dst.init(src.size_1 * src.size_2 * src.size_3);
	int x = 0;
	for (int i = 0; i < src.size_1; i++)
		for (int j = 0; j < src.size_2; j++)
			for (int k = 0; k < src.size_3; k++) {
				dst.data[x] = src.data[i][j][k];
				x++;
			}
}

void Flatten(array2D& src, array1D& dst) {
	dst.init(src.size_1 * src.size_2);
	int x = 0;
	for (int i = 0; i < src.size_1; i++)
		for (int j = 0; j < src.size_2; j++) {
			dst.data[x] = src.data[i][j];
			x++;
		}
}