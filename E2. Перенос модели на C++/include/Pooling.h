#pragma once
#include "Arrays.h"

/// <summary>
/// Входной массив в формате КАНАЛ/ШИРИНА/ВЫСОТА
/// Выходной массив в формате КАНАЛ/ШИРИНА/ВЫСОТА
/// </summary>
void MaxPooling2D(array3D& src, array3D& dst, int h, int w);

/// <summary>
/// Входной массив в формате КАНАЛ/ШИРИНА/ВЫСОТА
/// Выходной массив в формате КАНАЛ/ШИРИНА/ВЫСОТА
/// </summary>
void MinPooling2D(array3D& src, array3D& dst, int h, int w);
