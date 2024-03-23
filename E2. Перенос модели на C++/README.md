# Описание слоёв нейронных сетей на C++

Данный проект создан не столько с целью переноса сетей на язык C, сколько для того, чтобы показать, что "под капотом" большинства слоёв нейронных сетей находятся простейшие операции. Для удобства работы, были описаны 4 типа массивов (от одномерного до четырёхмерного). 

Пример реализации такого массива:
```C++
// Двумерный массив
struct array2D {
	float** data;	// Данные
	int size_1;     // Длина массива
	int size_2;     // Ширина массива

	// Инициализация массива указанного размера и выделение памяти
	void init(int size1, int size2) {
		size_1 = size1;
		size_2 = size2;
		data = new float* [size1];
		for (int i = 0; i < size1; i++)
			data[i] = new float[size2];
	}

	// Установка значения элементов массива
	void set(float c) {
		for (int i = 0; i < size_1; i++)
			for (int j = 0; j < size_2; j++)
				data[i][j] = c;
	}

	// Заполнение массива значениями 1
	void set_one() {
		set(1);
	}

	// Вывод массива на экран
	void print() {
		for (int i = 0; i < size_1; i++) {
			for (int j = 0; j < size_2; j++)
				std::cout << data[i] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
};
```

Для каждого типа слоёв есть свои реализации. 
Например, Dense:

```C++
void Dense(array1D& src, array1D& dst, array2D& weights, array1D& bias, int size) {
	dst.init(size);
	dst.set(0);
	for (int i = 0; i < size; i++)
		dst.data[i] = bias.data[i];

	for (int i = 0; i < src.size; i++)
		for (int j = 0; j < size; j++)
			dst.data[j] += src.data[i] * weights.data[j][i];
}
```

А также весь спектр активационных функций,
например, ELU:

```C++
float elu(float data) {
	if (data > 0) return data;
	return exp(data) - 1;
}
```

Из этих слоёв можно собрать сеть достаточно сложые сети, подтягивая из файла весовые коэффициенты, и, тем самым решить задачу построения нейронных сетей в обход любых SDK. Однако для формирования весов, в любом случае не обойтись без средств для обучения, посокльку построение своих оптимизаторов задача не из простых.


