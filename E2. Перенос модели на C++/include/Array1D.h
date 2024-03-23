#include <iostream>
#pragma once

// Одномерный массив
struct array1D {
	// Данные
	float* data;

	// Длина массива
	int size;

	// Инициализация массива указанного размера и выделение памяти
	void init(int size) {
		this->size = size;
		data = new float[size];
	}

	// Заполнение массива значениями 1
	void set_one() {
		set(1);
	}

	// Заполнения массива указанным числом
	void set(float c) {
		for (int i = 0; i < size; i++)
			data[i] = c;
	}

	// Вывод массива на экран
	void print() {
		for (int i = 0; i < size; i++)
			std::cout << data[i] << " ";
		std::cout << std::endl;
	}
};