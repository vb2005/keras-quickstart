#pragma once
#include "Arrays.h"

// Полносвязный слой без подстроечных значений
void Dense(array1D& src, array1D& dst, array2D& weights, int size);

// Полносвязный слой с подстройкой
void Dense(array1D& src, array1D& dst, array2D& weights, array1D& bias, int size);