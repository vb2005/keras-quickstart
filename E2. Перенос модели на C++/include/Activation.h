#pragma once
#include "Arrays.h"


float linear(float data);

float step(float data);

float elu(float data);

float relu(float data);

float sigmoid(float data);

float softSign(float data);

float silu(float data);

float softMax(array1D src, float data);

// Функция активации
void Activation(array3D& src, array3D& dst, float (*actType)(float));


