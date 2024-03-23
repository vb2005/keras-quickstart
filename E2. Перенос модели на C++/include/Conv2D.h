#pragma once
#include "Arrays.h"

/// <summary>
/// Входной массив в формате КАНАЛ/ШИРИНА/ВЫСОТА
/// Выходной массив в формате КАНАЛ/ШИРИНА/ВЫСОТА
/// Веса в формате ОКНО/КАНАЛ/ШИРИНА/ВЫСОТА
/// </summary>
void Conv2D(array3D& src, array3D& dst, array4D& weights, array1D& biases);

