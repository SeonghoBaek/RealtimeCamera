//
// Created by major on 17. 1. 5.
//
#include <stdio.h>
#include <math.h>
#include "Classifier.h"

float IClassifier::getL2Distance(Vector& v1, Vector& v2)
{
    int i = 0;
    float sum = 0.0;
    float *fv1 = (float *)v1.mpData;
    float *fv2 = (float *)v2.mpData;

    int vectorLength = v1.mLength / (sizeof(float));

    for (i = 0; i < vectorLength; i++)
    {
        float diff = fv1[i] - fv2[i];

        sum += pow(diff, 2);
    }

    sum = sqrt(sum);

    return sum;
}

float IClassifier::getEuclidianSimilarity(Vector& fv1, Vector& fv2)
{
    float distance = this->getL2Distance(fv1, fv2);
    float similarity = 1/(1 + distance);

    printf("simlarity: %f\n", similarity);

    return similarity;
}

float EuclidianDistanceClassifier::getSimilarity(Vector &v1, Vector &v2) {
    return this->getEuclidianSimilarity(v1, v2);
}

float EuclidianDistanceClassifier::getSVMClass(Vector &v) {
    return 0.0;
}

float EuclidianDistanceClassifier::getDNNClass(Vector &v) {
    return 0.0;
}
