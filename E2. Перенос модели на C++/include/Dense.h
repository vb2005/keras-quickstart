#pragma once
#include "Arrays.h"

// ������������ ���� ��� ������������ ��������
void Dense(array1D& src, array1D& dst, array2D& weights, int size);

// ������������ ���� � �����������
void Dense(array1D& src, array1D& dst, array2D& weights, array1D& bias, int size);