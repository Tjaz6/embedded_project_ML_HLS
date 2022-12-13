#include <stdio.h>
#include <iostream>
#include "matmul.hpp"

#include "/home/tjaz/final_ass/embedded_project_ML/test_images.h"

int main() {

	for(int i=1; i<19; i++){


	//int i =1;
		float temp_image[400]={0};

		for(int j=0;j<400;j++){
			temp_image[j]=images_array[i][j];
		}


		int pred = nn_inference(temp_image);
		std::cout << std::endl;
		std::cout << i << std::endl;
		std::cout << "NN Prediction: " << pred << std::endl;
		std::cout << std::endl;


	}




return 0;
}
