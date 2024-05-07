

#ifndef PA2_C00532580_LABELLEDDATASET_H
#define PA2_C00532580_LABELLEDDATASET_H
#include<iostream>
#include<exception>
#include<iomanip>
#include<sstream>
#include<stdexcept>
#include<fstream>
#include<string>
#include<vector>
class LabelledDataset{
public:
    std::vector<std::vector<double>> DataFeatures;
    std::vector<std::vector<double>> DataLabels;
    LabelledDataset(std::string);
};
#endif //PA2_C00532580_LABELLEDDATASET_H
