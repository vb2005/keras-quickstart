#include <iostream>
#pragma once

// Двумерный массив
struct array2D {
	// Данные
	float** data;

	// Длина массива
	int size_1;

	// Ширина массива
	int size_2;

	// Инициализация массива указанного размера и выделение памяти
	void init(int size1, int size2) {
		size_1 = size1;
		size_2 = size2;
		data = new float* [size1];
		for (int i = 0; i < size1; i++)
			data[i] = new float[size2];
	}

	// Установка значения элементов массива
	void set(float c) {
		for (int i = 0; i < size_1; i++)
			for (int j = 0; j < size_2; j++)
				data[i][j] = c;
	}

	// Заполнение массива значениями 1
	void set_one() {
		set(1);
	}

	// Вывод массива на экран
	void print() {
		for (int i = 0; i < size_1; i++) {
			for (int j = 0; j < size_2; j++)
				std::cout << data[i] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
};