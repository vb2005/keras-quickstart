#include <iostream>
#include "DNN.h"

int main() {
	// INPUT TENSOR
	array3D intput_0;
	// Значения
	intput_0.init(3, 4, 4);
	intput_0.set_one();
	//********************************************

	// CONV_1
	array3D conv_1_res;
	// Веса
	array4D conv_1_weights;
	array1D conv_1_bias;
	conv_1_bias.init(1);
	conv_1_bias.data[0] = 100;
	conv_1_weights.init(1, 3, 3, 3);
	conv_1_weights.set_one();
	//********************************************

	// ACT_1
	array3D act_1_res;
	//********************************************

	// POOL_1
	array3D pool_1_res;
	//********************************************

	// Flat_1
	array1D flat_1_res;
	//********************************************

	Conv2D(intput_0, conv_1_res, conv_1_weights, conv_1_bias);
	Activation(conv_1_res, act_1_res, &elu);
	Flatten(act_1_res, flat_1_res);
	//MaxPooling2D(act_1_res, pool_1_res, 2, 2);
	flat_1_res.print();
	std::system("pause");
}

