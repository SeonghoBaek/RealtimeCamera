#include "utils.hpp"
#include <iostream>

using namespace std;

// print usage information
inline void Usage(const char *exe) {
	cerr<< "Usage: "<< endl
		<< "  "<< exe<< "train_sample_list_file n_subject_samples "
		<< "sample_width sample_height SRC-model"<< endl;
}

int train(int ac, char **av) {

	const string train_sample_list_file= "../face_register/sparse/sparse.train_list";
	const size_t n_subject_samples= 30;
	const int sample_width= 10;
	const int sample_height= 12;
	CvSize sample_size= cvSize(sample_width, sample_height);
	const string src_model_file= "../face_register/sparse/sparse.src";

	vector<string> train_sample_list;
	LoadSampleList(train_sample_list_file, &train_sample_list);
	//assert(train_sample_list.size() % n_subject_samples == 0);

	SRCModel *src= TrainSRCModel(train_sample_list, sample_size, n_subject_samples);
	SaveSRCModel(src, src_model_file);
	ReleaseSRCModel(&src);	

	return 0;
}