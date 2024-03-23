#include <iostream>
#pragma once

// ���������� ������
struct array1D {
	// ������
	float* data;

	// ����� �������
	int size;

	// ������������� ������� ���������� ������� � ��������� ������
	void init(int size) {
		this->size = size;
		data = new float[size];
	}

	// ���������� ������� ���������� 1
	void set_one() {
		set(1);
	}

	// ���������� ������� ��������� ������
	void set(float c) {
		for (int i = 0; i < size; i++)
			data[i] = c;
	}

	// ����� ������� �� �����
	void print() {
		for (int i = 0; i < size; i++)
			std::cout << data[i] << " ";
		std::cout << std::endl;
	}
};