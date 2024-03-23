#pragma once
#include "Arrays.h"

/// <summary>
/// ������� ������ � ������� �����/������/������
/// �������� ������ � ������� �����/������/������
/// </summary>
void MaxPooling2D(array3D& src, array3D& dst, int h, int w);

/// <summary>
/// ������� ������ � ������� �����/������/������
/// �������� ������ � ������� �����/������/������
/// </summary>
void MinPooling2D(array3D& src, array3D& dst, int h, int w);
