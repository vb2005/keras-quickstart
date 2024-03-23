#include "Pooling.h"
#include "Utils.h"

void MaxPooling2D(array3D& src, array3D& dst, int h, int w) {
	int w_2 = src.size_2 / w;
	int h_2 = src.size_3 / h;
	dst.init(src.size_1, w_2, h_2);
	for (int c = 0; c < src.size_1; c++)
	{
		for (int x = 0; x < w_2; x++)
			for (int y = 0; y < h_2; y++)
			{
				float max_val = src.data[c][x * 2][y * 2];
				for (int i = 0; i < w; i++)
					for (int j = 0; j < h; j++)
						max_val = max(max_val, src.data[c][x * w + i][y * h + j]);
				dst.data[c][x][y] = max_val;
			}
	}
}

void MinPooling2D(array3D& src, array3D& dst, int h, int w) {
	int w_2 = src.size_2 / w;
	int h_2 = src.size_3 / h;
	dst.init(src.size_1, w_2, h_2);
	for (int c = 0; c < src.size_1; c++)
	{
		for (int x = 0; x < w_2; x++)
			for (int y = 0; y < h_2; y++)
			{
				float min_val = src.data[c][x * 2][y * 2];
				for (int i = 0; i < w; i++)
					for (int j = 0; j < h; j++)
						min_val = min(min_val, src.data[c][x * w + i][y * h + j]);
				dst.data[c][x][y] = min_val;
			}
	}
}