//
// Created by major on 17. 1. 6.
//

#ifndef REALTIMECAMERA_VECTORQUEUE_H
#define REALTIMECAMERA_VECTORQUEUE_H

#include "Common.h"
#include "Lock.h"

#define VECTOR_QUEUE_SIZE 64

enum
{
    ITEM_BYTE = 1,
    ITEM_SHORT,
    ITEM_INT,
    ITEM_FLOAT,
    ITEM_DOUBLE
};

class Vector
{
public:

    void *mpData;
    unsigned int mLength; // bytes
    int mFrame;    // reserved
    int mType;   // reseved: 1 - byte, 2 - shot, 3 - int, 4 - float, 5 - double
    int mX;
    int mY;
    int mT;
    int mB;
    float mConfidence;
    int mLabelIndex;

    Vector();
    Vector(unsigned int length);
    Vector(const Vector&);
    Vector& operator=(const Vector&);

    bool valid() { if (mpData == NULL) return FALSE; return TRUE; }

    virtual ~Vector();
};

class VectorQueue {
private:
    int mBottom;
    int mTop;
    int mQSize; // Total Q Size.
    int mSize;  // Num in Q
    bool mFull;

    Vector* mpQ;
    Mutex_t mMutex;

public:
    VectorQueue();
    VectorQueue(int size);

    virtual ~VectorQueue();
    void 		clear();
    int 		push(Vector& item);
    Vector*	    pop();
    int 		getSize();
    int 		getQSize() { return this->mQSize;}
};

class IVectorNotifier
{
public:
    virtual int nodtify(float userdata, Vector& vector) = 0;
};
#endif //REALTIMECAMERA_VECTORQUEUE_H
