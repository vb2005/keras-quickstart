#include "Conv2D.h"
void Conv2D(array3D& src, array3D& dst, array4D& weights, array1D& biases) {

	dst.init(weights.size_1, src.size_2, src.size_3);
	int offset_1 = (weights.size_3 - 1) / 2;
	int offset_2 = (weights.size_4 - 1) / 2;

	// Для каждого окна свертки
	for (int ww = 0; ww < weights.size_1; ww++)
	{
		for (int x = 0; x < src.size_2; x++)
		{
			for (int y = 0; y < src.size_3; y++)
			{
				dst.data[ww][x][y] = 0;

				// Если изображение было многоканальным, 
				// то мы своё окно свёртки примняем для кажого канала
				for (int cc = 0; cc < src.size_1; cc++)
				{
					float sum = 0;
					for (int i = -offset_1; i <= offset_1; i++)
						for (int j = -offset_2; j <= offset_2; j++)
						{
							if (x + i < 0) continue;
							if (y + j < 0) continue;
							if (x + i >= src.size_2) continue;
							if (y + j >= src.size_3) continue;
							float f = src.data[cc][x + i][y + j];
							sum += f * weights.data[ww][cc][i + offset_1][j + offset_2];
						}
					dst.data[ww][x][y] += sum;
				}
				dst.data[ww][x][y] += biases.data[ww];
			}
		}
	}
}