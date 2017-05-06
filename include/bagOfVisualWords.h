/*
 *Created by Deangeli Gomes Neves
 *
 * This software may be freely redistributed under the terms
 * of the MIT license.
 *
 */

#ifndef _BAGOFVISUALWORDS_H_
#define _BAGOFVISUALWORDS_H_

#include "common.h"
#include "featureVector.h"
#include "file.h"
#include "image.h"
#include "histogram.h"


typedef struct _bagOfVisualWords {
    int patchSize;
    DirectoryManager* directoryManager;
    FeatureMatrix* vocabulary;
} BagOfVisualWords;

typedef struct _vocabularyTraining {
    int nlabels;
    int* labels;
    FeatureMatrix* dictionary;
} VocabularyTraining;

typedef struct _trainingKnowledge{
    int nlabels;
    int nvocabulary;
    int* labels;
    int** imageHistograms;//[nlabels, nvocabulary]
}TrainingKnowledge;


FeatureMatrix* computeFeatureVectors(Image* imagePack, int patchSize);
FeatureMatrix* computeFeatureVectors(DirectoryManager* directoryManager, int patchSize);
VocabularyTraining* kMeansClustering(FeatureMatrix* featureMatrix, int numberOfCluster);
TrainingKnowledge* createTrainingKnowledge(int numberOfImages, int vocabularySize);
TrainingKnowledge* trainWithImage(int k, Image* image, TrainingKnowledge* trainingKnowledge,
    VocabularyTraining* vocabularyTraining);
float euclidean_distance(int n, int* v0, int* v1);
void findLabels (TrainingKnowledge* trainingKnowledge,TrainingKnowledge* testKnowledge);



//delete functions for these structures



#endif //LIBFL_BAGOFVISUALWORDS_H
