#include "utils.hpp"
#include <iostream>
#include <opencv2/core/types_c.h>

using namespace std;

SRCModel *gSRCModel = NULL;

// print usage information
inline void Usage(const char *exe)
{
	cerr<< "Usage: "<< endl
		<< "  "<< exe<< "SRC-model test-sample-list-file SCI-threshold"<< endl;
}

void init_src_model()
{
	const string src_model_file= "../face_register/sparse/sparse.src";

	if (gSRCModel == NULL)
	{
        cout << "Load SRC Model" << endl;
		gSRCModel = LoadSRCModel(src_model_file);
	}
}

int src_test_file(const char *img_file_path, const char* label)
{

    const double sci_t= 0;
    int res = -1;

	if (gSRCModel == NULL) return 0; // Model Not Exist or Load Fail. Use only svm.

    SRCModel *src_model= gSRCModel;
	string img = img_file_path;

    //cout << "sample w: " << src_model->sample_size_.width << ", h: " << src_model->sample_size_.height << endl;
    CvMat *y= LoadSample(img_file_path, src_model->sample_size_);

    string name= Recognize(src_model, y, sci_t, NULL, NULL);

	string l = label;

    cout << "T: " << l << ", S: " << name << endl;

	if (l == name)
	{
		res = 0;
	}

    cvReleaseMat(&y);

    //ReleaseSRCModel(&src_model);

    return res;
}

int test(int ac, char **av) {
	const string src_model_file= "../face_register/sparse/sparse.src";
	const string test_sample_list_file= "../face_register/sparse/sparse.test_list";
	const double sci_t= 0;

	vector<string> test_sample_list;
	LoadSampleList(test_sample_list_file, &test_sample_list);

	SRCModel *src_model= LoadSRCModel(src_model_file);

	int ok_cnt= 0;
	for(size_t i=0; i<test_sample_list.size(); ++i)
	{
		CvMat *y= LoadSample(test_sample_list[i], src_model->sample_size_);
		string name= Recognize(src_model, y, sci_t, 
			(test_sample_list[i]+".x").c_str(), (test_sample_list[i]+".r").c_str());
		cout<< test_sample_list[i]<< " " << Filename2ID(test_sample_list[i])<< " " << name<< endl;
		if(Filename2ID(test_sample_list[i]) == name) {
			++ok_cnt;
		}
		cvReleaseMat(&y);
	}

	cout<< "ok count : "<< ok_cnt<< endl
		<< "total count : "<< test_sample_list.size()<< endl
		<< "precision : "<< double(ok_cnt)/test_sample_list.size()<< endl;

	ReleaseSRCModel(&src_model);

	return 0;
}