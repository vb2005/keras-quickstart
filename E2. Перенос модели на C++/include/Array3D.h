#include <iostream>
#pragma once

// Трёхмерный массив
struct array3D {

	// Данные
	float*** data;

	// Размерности
	int size_1;
	int size_2;
	int size_3;

	// Инициализация массива указанного размера и выделение памяти
	void init(int size1, int size2, int size3) {
		size_1 = size1;
		size_2 = size2;
		size_3 = size3;
		data = new float** [size1];
		for (int i = 0; i < size1; i++) {
			data[i] = new float* [size2];
			for (int j = 0; j < size2; j++)
				data[i][j] = new float[size3];
		}
	}

	// Установка значения элементов массива
	void set(float c) {
		for (int i = 0; i < size_1; i++)
			for (int j = 0; j < size_2; j++)
				for (int k = 0; k < size_3; k++)
					data[i][j][k] = c;
	}

	// Заполнение массива значениями 1
	void set_one() {
		set(1);
	}

	// Вывод массива на экран
	void print() {
		for (int i = 0; i < size_1; i++)
		{
			for (int j = 0; j < size_2; j++)
			{
				for (int k = 0; k < size_3; k++)
					std::cout << data[i][j][k] << " ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
};
