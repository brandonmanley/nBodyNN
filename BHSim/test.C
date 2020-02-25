#include <stdio.h> 
#include <string.h> 
#include <stdlib.h>
#include <iostream>
#include <vector>

int main() {

    int buffSize = 256;
    FILE *fp;
    char buff[buffSize];
    int numLines = 100;

    fp = fopen("/nBodyData/inputs/indat_3_1.dat", "r");

    for(int j=0; j< numLines; ++j){
        fgets(buff, buffSize, (FILE*)fp);
        // printf("%s\n", buff);

        char *token = std::strtok(buff, ",");
        std::vector<double> initialParameters;
        while (token != NULL){ 

            char *ptr;
            double ret = 0.0;
            ret = std::strtod(token, &ptr);
            initialParameters.push_back(ret);
            token = std::strtok(NULL, ","); 
        }
        for(int i=0; i< initialParameters.size(); ++i){
            std::cout << initialParameters[i] << " ";
        }
        std::cout << std::endl;

    }
    fclose(fp);
    return 0;
}