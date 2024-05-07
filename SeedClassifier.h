#ifndef PA2_C00532580_SEEDCLASSIFIER_H
#define PA2_C00532580_SEEDCLASSIFIER_H
#include<vector>
#include<cmath>
class SeedClassifier{
public:
    std::vector<std::vector<double>> W1, W2;
    double AverageError;
    SeedClassifier();
    double Score(std::vector<double> SampleOutput,
                 std::vector<double> SampleLabel);
    std::vector<double> operator<<(std::vector<double> Sample);

};
#endif //PA2_C00532580_SEEDCLASSIFIER_H
