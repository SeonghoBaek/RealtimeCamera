//
// Created by major on 17. 1. 5.
//

#ifndef REALTIMECAMERA_CLASSIFIER_H
#define REALTIMECAMERA_CLASSIFIER_H

#include "VectorQueue.h"

typedef enum
{
    EUCLIDIAN_SIMILARITY,
    SVM_PROBABILITY,
    DNN_PROBABILITY
} ClassifierT;

// classifier = ClassifierFactory.instance(EUCLIDIAN_SIMILARITY);
// classifier.evaluate(

class IClassifier
{
private:
    float getL2Distance(Vector& fv1, Vector& fv2);

public:
    IClassifier() {}
    virtual ~IClassifier() {}

    float getEuclidianSimilarity(Vector& fv1, Vector& fv2);

    float getSVMProbability(Vector& fv)
    {
        // TO DO.
        return 1.0;
    }

    float getNNProbability(Vector& fv)
    {
        // TO DO
        return 1.0;
    }

    virtual float getSimilarity(Vector& v1, Vector& v2) = 0;
    virtual float getSVMClass(Vector& v) = 0;
    virtual float getDNNClass(Vector& v) = 0;
};

class EuclidianDistanceClassifier: public IClassifier
{
public:
    EuclidianDistanceClassifier() {}
    virtual ~EuclidianDistanceClassifier() {}

    float getSimilarity(Vector &v1, Vector &v2) override;

    float getSVMClass(Vector &v) override;

    float getDNNClass(Vector &v) override;
};

class ClassifierFactory
{
public:
    static IClassifier* build(ClassifierT type)
    {
        if (type == EUCLIDIAN_SIMILARITY)
        {
            return new EuclidianDistanceClassifier();
        }
    }
};

#endif //REALTIMECAMERA_CLASSIFIER_H
