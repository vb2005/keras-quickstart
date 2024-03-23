#include <iostream>
#pragma once

// ��������� ������
struct array2D {
	// ������
	float** data;

	// ����� �������
	int size_1;

	// ������ �������
	int size_2;

	// ������������� ������� ���������� ������� � ��������� ������
	void init(int size1, int size2) {
		size_1 = size1;
		size_2 = size2;
		data = new float* [size1];
		for (int i = 0; i < size1; i++)
			data[i] = new float[size2];
	}

	// ��������� �������� ��������� �������
	void set(float c) {
		for (int i = 0; i < size_1; i++)
			for (int j = 0; j < size_2; j++)
				data[i][j] = c;
	}

	// ���������� ������� ���������� 1
	void set_one() {
		set(1);
	}

	// ����� ������� �� �����
	void print() {
		for (int i = 0; i < size_1; i++) {
			for (int j = 0; j < size_2; j++)
				std::cout << data[i] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
};