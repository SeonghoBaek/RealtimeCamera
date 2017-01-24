//
// Created by major on 17. 1. 11.
//

#ifndef FACE_DETECTOR_IDENTIFIER_H
#define FACE_DETECTOR_IDENTIFIER_H
#ifdef __cplusplus
extern "C" {
#endif

int run_identifier();
void clear_label_check_info();
char* get_label_in_box(int x, int y, int h, int w);
void version_up();
#ifdef USE_SRC
void train_sparse();
int test_image_file(const char *image_file_path, const char *label);
#endif
#ifdef __cplusplus
}
#endif
#endif //FACE_DETECTOR_IDENTIFIER_H
