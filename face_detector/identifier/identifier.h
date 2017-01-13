//
// Created by major on 17. 1. 11.
//

#ifndef FACE_DETECTOR_IDENTIFIER_H
#define FACE_DETECTOR_IDENTIFIER_H
#ifdef __cplusplus
extern "C" {
#endif

int run_identifier();

char* get_lable_in_box(int x, int y, int h, int w);
void version_up();

#ifdef __cplusplus
}
#endif
#endif //FACE_DETECTOR_IDENTIFIER_H
