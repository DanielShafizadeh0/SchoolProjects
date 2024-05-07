#include"SeedClassifier.h"
/*Default SeedClassifier Constructor.
 * Initializes W1 and W2 of the Object to 2D double vectors with dimensions 7x21 and 21x3 respectively,
 * with random values between 0. and 1.0
 * */
SeedClassifier::SeedClassifier() {
W1.resize(7,std::vector<double> (21,((double)rand())/((double)(RAND_MAX+1.0))));
W2.resize(21,std::vector<double> (3,((double)rand())/((double)(RAND_MAX+1.0))));
}
/* Gives the Mean Squared Error score for the sample.
 * Given a sample prediction and the label for a sample, both double vectors of the same length,
 * the error is computed by squaring the difference of the Label's value at i
 * and the prediction (SampleOutput) at i, summing the values of these squared differences,
 * then dividing it by 3 (the sample/prediction length)
 * Input: a double vector representing the prediction and a double vector representing the label
 * Output: a double representing the mean squared error
 * */
double SeedClassifier::Score(std::vector<double> SampleOutput, std::vector<double> SampleLabel) {
double MSE=0.0;
    for(int i=0;i<SampleOutput.size();i++){
        MSE+= pow((SampleLabel[i]-SampleOutput[i]),2);
    }
    MSE=MSE/((double)SampleOutput.size());
    return MSE;
}
/*A function that overloads the insertion operator for SeedClassifier objects, allowing a
 * prediction for a sample to be produced via matrix multiplication.
 * Given a 1x7 vector of doubles, the matrix multiplication product with W1, a 7x21 2D double
 * vector, to produce a 1x21 2D double vector functioning like a matrix,
 * of which each value is filtered with a sigmoid activation
 * function, in the form of 1/(1+e^(-x)), where x is the value at the index (i,j),
 * in this case being the first dimension is always 1 in the product matrices means
 * i is always 0. The same principles are applied to the product of the first product matrix
 * and the W2 matrix (21x3), to produce a 1x3 prediction.
 * Input: a 1x7 double vector representing a sample
 * Output: A 1x3 double vector representing the prediction for the sample on the given
 * SeedClassifier Object
 * */
std::vector<double> SeedClassifier::operator<<(std::vector<double> Sample){
    //1x7 sample, 7x21 w1, 21x3 w2
    std::vector<std::vector<double>> actm1;
    actm1.resize(1,std::vector<double> (21));
    for(int j=0;j<this->W1[0].size();j++){
        double t=0.0;
        for(int k=0;k<this->W1.size();k++){
            t+= Sample[k]*this->W1[k][j];
        }
        actm1[0][j]=t;
        actm1[0][j]=1.0/(1.0+exp(-actm1[0][j]));
    }
    std::vector<std::vector<double>> actm2(21,std::vector<double>(3));
    for(int j=0;j<this->W2[0].size();j++){
        double t=0.0;
        for(int k=0;k<this->W2.size();k++){
            t+=actm1[0][k]*this->W2[k][j];
        }
        actm2[0][j]=t;
        actm2[0][j]=1.0/(1.0+exp(-actm2[0][j]));
    }


    return actm2[0];

}
