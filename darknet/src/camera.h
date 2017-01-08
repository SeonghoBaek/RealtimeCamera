#ifndef CAMERA
#define CAMERA

#include "image.h"
void detect_camera(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix);

#endif