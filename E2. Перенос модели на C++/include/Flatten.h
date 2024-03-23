#pragma once
#include "Arrays.h"

// 4D -> 1D
void Flatten(array4D& src, array1D& dst);

// 3D -> 1D
void Flatten(array3D& src, array1D& dst);

// 2D -> 1D
void Flatten(array2D& src, array1D& dst);
