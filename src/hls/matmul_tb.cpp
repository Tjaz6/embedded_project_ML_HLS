#include <stdio.h>
#include <iostream>
#include "matmul.hpp"

#include "/home/tjaz/final_ass/embedded_project_ML/test_images.h"

int main() {

	for(int i=0; i<20; i++){

		int pred = nn_inference(images_array[i]);
		std::cout << std::endl;
		std::cout << i << std::endl;
		std::cout << "NN Prediction: " << pred << std::endl;
		std::cout << std::endl;


	}




return 0;
}
