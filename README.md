# Simulating Genetics with Machine Learning
This project involves taking a dataset of wheat seed genetic data and use machine learning principles
to simulate generations of seed offspring until an ideal generation was produced.

The process involves taking the given sample and classifying them 
based on their label, creating a prediction of
the label via matrix multiplication and comparing it to the original 
to determine the Mean Squared error, then using the
mean squared error for each sample to determine the 
population error, which then serves as the basis
for the simulation iterations.

## LabelledDataset
This class takes a given file of sample data to create 210 objects with.
The first 7 values are sample features used in calculations with the last
value being the seed class. A file reader opens the specified file and uses
stringstream to parse through, taking the values and inserting them into 
a size 7 vector that is inserted into an object containing the data features
after it's full, and uses the class value to put a size 3 vector with a specific
set of values to put in the object containing data labels.
```C++
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
```
## SeedClassifier
This class provides the functions required to perform the operations for
evolving the samples. 
### SeedClassifier Constructor
The constructor sets the W1 and W2 matrices to random values.
W1 and W2 are double two-dimensional matrices with sizes 7x21 and 21x3 respectively, which 
are used with the remaining functions.
```C++
SeedClassifier::SeedClassifier() {
W1.resize(7,std::vector<double> (21,((double)rand())/((double)(RAND_MAX+1.0))));
W2.resize(21,std::vector<double> (3,((double)rand())/((double)(RAND_MAX+1.0))));
}
```
### Score
A function that takes in parameters of the prediction for a sample and the sample label (both length 3 vectors)
and gives the average mean squared error of the prediction. The function
subtracts the current sample prediction value from the current sample label value, squares the difference, then puts the
result in a sum that is divided by 3 to obtain the average.
```C++
double SeedClassifier::Score(std::vector<double> SampleOutput, std::vector<double> SampleLabel) {
double MSE=0.0;
    for(int i=0;i<SampleOutput.size();i++){
        MSE+= pow((SampleLabel[i]-SampleOutput[i]),2);
    }
    MSE=MSE/((double)SampleOutput.size());
    return MSE;
}
```
### operator<<
The function overrides the << operator to allow SeedClassifier objects to
produce a  size 3 vector  representing a prediction for a given sample, a 
vector of length 7. Using the W1 and W2 weights, matrix multiplication is performed using the sample prediction and W1
to obtain a size 21 vector. The values obtained from multiplying are then converted with the function:

<img src="https://latex.codecogs.com/svg.image?\frac{1}{1&plus;e^{-x}}" title="\frac{1}{1+e^{-x}}"  alt="eq1"/>

The same process is repeated with W2 to obtain a size 3 vector with matrix multiplication that serves as the prediction,
which is then returned.

```c++
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
```
## main
The main class contains functions that use SeedClassifier objects and functions to produce a
prediction with the best SeedClassifier object from the best SeedClassifier population.

### Select
The Select method takes a vector of SeedClassifier objects, creates a new vector, then stores the first 10 
SeedClassifier objects in the newly created vector, finally adding 5 new SeedClassifier objects to the vector before returning it.

```c++
std::vector<SeedClassifier> Select(std::vector<SeedClassifier>& Objects){
    std::vector<SeedClassifier> best(50);

for(int i=0;i<10;i++){
best[i]=SeedClassifier(Objects[i]);
}
for(int i=0;i<5;i++){
    best[10+i]=SeedClassifier();
}
return best;
}
```
### Crossover
This method adds 15 SeedClassifier objects to the parameter vector.
Two random numbers between 0 and 9 are generated, corresponding to the
first 10 objects of the vector. If the first number is greater than the second, a copy of 
the corresponding SeedClassifier objects swap their W1 weights, and if less,
the copies swap their W2 weights. A copy of one of the objects is then inserted into the
parameter vector.
```c++
void Crossover(std::vector<SeedClassifier>& Objects){
    int rand1;
    int rand2;
    for(int i=0;i<15;i++) {
        rand1 = rand() % 10;
        rand2 = rand() % 10;
        while (rand1 == rand2) {
            rand1 = rand() % 10;
        }
        SeedClassifier ref1 = SeedClassifier(Objects[rand1]);
        SeedClassifier ref2 = SeedClassifier(Objects[rand2]);
        SeedClassifier refTemp=SeedClassifier(ref1);
        if (rand1 > rand2) {
            ref1.W1 = ref2.W1;
            ref2.W1 = refTemp.W1;
            refTemp=SeedClassifier(ref1);
        } else {
            ref1.W2 = ref2.W2;
            ref2.W2 = refTemp.W2;
            refTemp=SeedClassifier(ref2);
        }
        Objects[i+15]=SeedClassifier(refTemp);
    }
}
```
### Mutate
This method copies the SeedClassifier object randomly modifies the weights of the first 20 SeedClassifier objects in the
vector parameter, then adds them to the vector parameter. For each of the values in
W1 and W2, a random number out of 100 is generated, and if this number is less than 5, the value is incremented by
0.1, and decremented by 0.1 if it is greater than 4 but less than 10. After these operations are performed on each
object copy, the object is inserted into the vector parameter.
```c++
void Mutate(std::vector<SeedClassifier>& Objects){
SeedClassifier mutated;
int randi;
for(int i=0;i<20;i++){
    mutated=SeedClassifier(Objects[i]);
    for(int j=0;j<mutated.W1.size();j++){
        for(int l=0;l<mutated.W1[0].size();l++){
            randi=rand()%100;
            if(randi<5){
                mutated.W1[j][l]+=0.1;
            }
            else if(randi<10){
                mutated.W1[j][l]-=0.1;
            }
        }
    }
    for(int o=0;o<mutated.W2.size();o++){
        for(int p=0;p<mutated.W2[0].size();p++){
            randi=rand()%100;
            if(randi<5){
                mutated.W2[o][p]+=0.1;
            }
            else if(randi<10){
                mutated.W2[o][p]-=0.1;
            }
        }
    }
Objects[30+i]=mutated;
}
}
```
### int main()
The main method obtains the samples from the file then performs operations on them with the classes and methods
previously defined to simulate the offspring and produce an ideal generation.
The main function calls the LabelledDataset Constructor with the relative filepath to the Samples as the parameter,
handling any errors that might occur there. If it successfully executes, the features and the class for each entry are
printed in a list format. 
```c++
try{
    LabelledDataset theDataset = LabelledDataset("../cmake-build-debug/SeedClassificationData.txt");
        for (int i = 0; i < theDataset.DataFeatures.size(); i++) {
            std::cout << i + 1 << ". ";
            for (int j = 0; j < theDataset.DataFeatures[0].size(); j++) {
                std::cout << std::fixed << std::setprecision(3) << theDataset.DataFeatures[i][j] << " ";
            }
            if (theDataset.DataLabels[i][0] == 1.0) {
                std::cout << "Class: " << 0 << std::endl;
            }
            if (theDataset.DataLabels[i][1] == 1.0) {
                std::cout << "Class: " << 1 << std::endl;
            }
            if (theDataset.DataLabels[i][2] == 1.0) {
                std::cout << "Class: " << 2 << std::endl;
            }


        }
```
After creating and filling a SeedClassifier vector of size 50 named pop, an initial generation is created and has its
average population error printed. This is done by iterating through the pop variable, getting a prediction using the 
overloaded << operator on the SeedClassifier object with the Data features and using the return value in the score 
function with the sample prediction, which then sets the AverageError for the object and after iterating through, gets 
the population average. After printing out the initial population error, the objects are then sorted. The same steps are
then repeated within a loop until the population average error is less than 0.175.
```c++
 do{
            c=0.0;
            pop=Select(pop);
            Crossover(pop);
            Mutate(pop);
            for (int i = 0; i < pop.size(); i++) {
                double t=0.0;
                for (int j = 0; j < theDataset.DataFeatures.size(); j++) {
                    t+= pop[i].Score(pop[i]<<theDataset.DataFeatures[j],theDataset.DataLabels[j]);
                }
                t=t/210.0;
                pop[i].AverageError=t;
                c+=t;
            }
            c=c/50.0;
            std::sort(pop.begin(),pop.end(),[](SeedClassifier a,SeedClassifier b){return a.AverageError<b.AverageError;});
            std::cout<<co<<" - Population Average Error: "<<std::fixed << std::setprecision(6)<<c<<std::endl;
            co++;
        }while(c>=0.175);
```
Upon reaching the terminating condition, the population average error is printed along with the average error of the 
best performing SeedClassifier object. Upon doing so, a text file is created with predictions for each sample entry
and the actual label value.
```c++
std::cout<<"Population average score: "<<std::fixed<<std::setprecision(6)<<c<<std::endl;
        std::cout<<"Best average error: "<<std::fixed<<std::setprecision(6)<<pop[0].AverageError<<std::endl;

        //Writes to Predictions.txt the Predicted values and label values for the dataset entries,
        //the predictions using the << operator on the best performing SeedClassifier
        std::ofstream File("Predictions.txt");
        File<<"Daniel Shafizadeh \nXXXXXXXXX\ni\tP0\tP1\tP2\tL0\tL1\tL2"<<std::endl;
        for(int i=0;i<theDataset.DataLabels.size();i++){
            std::vector<double> temp=pop[0]<<theDataset.DataFeatures[i];
            File<<i+1<<"\t";
            File<<std::fixed;
            File<<std::setprecision(2);
            File<<temp[0]<<"\t"<<temp[1]<<"\t"<<temp[2]<<"\t"<<theDataset.DataLabels[i][0]<<"\t"<<theDataset.DataLabels[i][1]<<"\t"<<theDataset.DataLabels[i][2]<<std::endl;
        }
        File.close();
```