#include"LabelledDataset.h"
/*Creates a labelled Dataset object via a file.
 * If the file is able to be read, a stringsteam object parses the values using the comma and
 * newline characters as delimiters, putting the first seven values on the line into a
 * vector object to be inserted into the DataFeatures variable as an object. The eighth value
 * on the line contains the seed class of the object, which after being read from the stringstream object,
 * adds a DataLabel object corresponding to the class given to the DataLabel variable.
 * Closes the file afterwards. In the case the file is unable to be opened, an error is thrown
 * Input: A string corresponding to a file
 * */
LabelledDataset::LabelledDataset(std::string relFilename){
std::ifstream fileReader;
fileReader.open(relFilename);
if(fileReader.fail()){
    try{
        throw std::ios::failure("Error: The specified file cannot be found.");
    }
    catch(std::ios::failure e){
        std::cout<<e.what();
    }
}
else{
    std::stringstream parser;
    std::string line;
    while(std::getline(fileReader,line)){
        parser=std::stringstream((line));
        std::vector<double> featureVals(7);
        for(int i=0;i<7;i++){
            std::string temp;
           std::getline(parser,temp,',');
            featureVals[i]=std::stod(temp);
        }
        DataFeatures.push_back(featureVals);
        std::string tmp;
        std::getline(parser,tmp,'\n');
        if(tmp=="Rosa"){
            DataLabels.push_back({1.0,0.0,0.0});
        }
        if(tmp=="Kama"){
            DataLabels.push_back({0.0,1.0,0.0});
        }
        if(tmp=="Canadian"){
            DataLabels.push_back({0.0,0.0,1.0});
        }




    }

}
    fileReader.close();
}

