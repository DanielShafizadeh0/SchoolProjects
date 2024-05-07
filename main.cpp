#include <iostream>
#include<exception>
#include<iomanip>
#include<vector>
#include<algorithm>
#include"LabelledDataset.h"
#include"SeedClassifier.h"
/* Creates a vector of 15 SeedClassifier objects given a vector of SeedClassifiers.
 * Given a vector of SeedClassifier Objects, the first 10 are inserted into a new
 * SeedClassifier vector, with 5 random ones added afterwards, then returns the
 * new vector object.
 * Input: A vector of SeedClassifier Objects
 * Output: A new vector of 15 SeedClassifier Objects, 10 from the input vector and 5 random
 * ones
 * */
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
/* Adds 15 new objects to the given SeedClassifier Vector.
 * Two distinct random integers are used select two SeedClassifier objects from the first 10 of the
 * input vector via indexes, creates copies of them, then depending on which number is larger,
 * swaps the W1 and W2 variables of the objects, then proceeds to add one of the object copies
 * to the input vector. 15 total objects are added.
 * Input: a vector of SeedClassifier Objects
 * */
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
/*Randomly modifies the weights in the first 20 objects in the input vector
 * For each of the 20 objects in the input vector, the W1 and W2 2D vector variables are
 * iterated over, and modified based on a random integer value. If the random integer is
 * less than 5, the value in the 2D vector variable is incremented by 0.1, and decremented
 * by 0.1 if  the integer is greater than 4 but less than 10. Afterwards, the object is added
 * to the end of the input vector.
 * Input: A vector of SeedClassifier Objects
 * */
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

int main() {
    /*Attempts to create a dataset object with the given SeedClassificationData file,
     * then prints out the values and class per line*/
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
        /* Creates a vector of SeedClassifier objects with size 50, filling it with SeedClassifier
         * variables*/
        std::vector<SeedClassifier> pop(50);
        for (int i = 0; i < 50; i++) {
            pop[i]=SeedClassifier();
        }
        /* Gets the Average Error for each SeedClassifier Object in the initial population by
         * summing the error scores of each dataset label, after obtaining the predicted
         * label with the << operator on the given SeedClassifier object using the weights of
         * the dataset entry.
         * */
        double c=0.0;
        for (int i = 0; i < pop.size(); i++) {
            double t=0.0;
            for (int j = 0; j < theDataset.DataFeatures.size(); j++) {
                t+= pop[i].Score(pop[i]<<theDataset.DataFeatures[j],theDataset.DataLabels[j]);
            }
            t=t/210.0;
            pop[i].AverageError=t;
            c+=t;

        }
        //Gives the average population error, prints it, then sorts the population in ascending
        //order of average error
        c=c/50.0;
        std::cout<<"Average Initial Population Error: "<<c<<std::endl;
        std::sort(pop.begin(),pop.end(),[](SeedClassifier a,SeedClassifier b){return a.AverageError<b.AverageError;});
        int co=1;
        /*A loop that applies the select, crossover, and mutate methods to the SeedClassifier population,
         * sets the new average error scores for the object, and the population as a whole,
         * then sorts the population in ascending order before evaluating the terminating
         * condition, the condition being when the population average error is less than 0.175.
         * The sort ensures the objects selected from the population are the best performing
         * (having the lowest error scores).
         * */
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
        //Prints the Population average error score of the best given population along with
        // the object with the best average error
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
    }catch (const std::exception& e){
        std::cout<<"Exception: "<<e.what()<<std::endl;
    }
    return 0;
}
