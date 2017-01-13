#include "Log.h"
#include "VectorQueue.h"

Vector::Vector()
{
    this->mpData = NULL;
    this->mLength = 0;
    this->mFrame = 0;
    this->mType = ITEM_FLOAT;
    this->mX = -1;
    this->mY = -1;
    this->mT = -1;
    this->mB = -1;
    this->mConfidence = 0.0;
    this->mLabelIndex = -1;
}

Vector::Vector(unsigned int length)
{
    if (length)
    {
        this->mpData = new char[length];
        memset(this->mpData, 0, length);
    }
    else
    {
        this->mpData = NULL;
    }

    this->mLength = length;
    this->mFrame = 0;
    this->mType = ITEM_FLOAT;
    this->mX = -1;
    this->mY = -1;
    this->mT = -1;
    this->mB = -1;
    this->mConfidence = 0.0;
    this->mLabelIndex = -1;
}

Vector::~Vector()
{
    if (this->mpData)
    {
        //LOGI("Delete mpData");
        delete [] (char *)mpData;
    }
}

Vector::Vector(const Vector& n)
{
    this->mFrame = n.mFrame;
    this->mLength = n.mLength;

    if (n.mLength)
    {
        this->mpData = new char[n.mLength];
        memset(this->mpData, 0, n.mLength);
        memcpy((char *)this->mpData, (char *)n.mpData, n.mLength);
    }
    else
    {
        this->mpData = NULL;
    }

    this->mType = n.mType;
    this->mX = n.mX;
    this->mY = n.mY;
    this->mT = n.mT;
    this->mB = n.mB;
    this->mConfidence = n.mConfidence;
    this->mLabelIndex = n.mLabelIndex;
}

Vector& Vector::operator=(const Vector& n)
{
    if (this != &n)
    {
        if (this->mpData) delete [] (char *)this->mpData;

        this->mpData = NULL;
        this->mFrame = n.mFrame;
        this->mLength = n.mLength;

        if (n.mLength)
        {
            this->mpData = new char[n.mLength];
            memcpy((char *)this->mpData, (char *)n.mpData, n.mLength);
        }

        this->mType = n.mType;
        this->mX = n.mX;
        this->mY = n.mY;
        this->mT = n.mT;
        this->mB = n.mB;
        this->mConfidence = n.mConfidence;
        this->mLabelIndex = n.mLabelIndex;
    }

    return *this;
}

VectorQueue::VectorQueue()
{
    this->mBottom = 0;
    this->mQSize = VECTOR_QUEUE_SIZE;
    this->mSize = 0;
    this->mTop = 0;
    this->mpQ = new Vector[VECTOR_QUEUE_SIZE];
    this->mMutex = Lock::createMutex();
    this->mFull = FALSE;
}

VectorQueue::VectorQueue(int size)
{
    this->mBottom = 0;
    this->mQSize = 0;
    this->mSize = 0;
    this->mQSize = size;
    this->mTop = 0;
    this->mpQ = NULL;
    this->mpQ = new Vector[size];
    this->mMutex = Lock::createMutex();
    this->mFull = FALSE;
}

VectorQueue::~VectorQueue()
{
    //LOGI("Delete QUEUE");
    for (int i = 0; i < this->mQSize; i++)
    {
        if (this->mpQ[i].mpData) delete [] (char *)this->mpQ[i].mpData;
        this->mpQ[i].mpData = NULL;
    }

    Lock::deleteMutex(this->mMutex);

    delete [] this->mpQ;
}

void VectorQueue::clear()
{
    int numQ = 0;

    LOCK(mMutex)
    {
        for (numQ = 0; numQ < mQSize; numQ++)
        {
            if (mpQ[numQ].mpData != NULL)
            {
                delete [] (char *)mpQ[numQ].mpData;

                mpQ[numQ].mpData = NULL;
            }
        }

        mSize = 0;
        mTop = 0;
        mBottom = 0;
        mFull = FALSE;
    }
}

int VectorQueue::push(Vector& item)
{
    int nr_q = 0;

    LOCK(mMutex)
    {
        if (mFull == TRUE)
        {
            nr_q = mQSize;
            LOGW("Queue Full");
        }
        else
        {
            mpQ[mTop] = item;

            mTop++;
            mTop %= mQSize;
            mSize++;

            nr_q = mSize;

            if (mSize == mQSize)
            {
                mFull = TRUE;
            }
        }
    }

    return nr_q;
}

Vector* VectorQueue::pop()
{
    Vector *item = new Vector;

    LOCK(mMutex)
    {
        if (this->mSize == 0) // empty.
        {
            LOGW("Queue Empty");
        }
        else
        {
            *item = mpQ[mBottom];

            if (mpQ[mBottom].mpData)
            {
                delete [] (char *)mpQ[mBottom].mpData;
                mpQ[mBottom].mpData = NULL;
            }

            mBottom++;
            mBottom %= mQSize;
            mSize--;
            mFull = FALSE;
        }
    }

    return item;
}

int VectorQueue::getSize()
{
    int size = 0;

    LOCK(mMutex)
    {
        size = mSize;
    }

    return size;
}