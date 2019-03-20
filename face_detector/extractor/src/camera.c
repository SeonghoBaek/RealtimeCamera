#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "camera.h"
#include <sys/time.h>

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
#endif

#ifdef REDIS
#include <hiredis.h>
#include <unistd.h>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/highgui/highgui_c.h>

#endif

#include "identifier.h"

/*
 * Update wait time. Added by Seongho Baek 2019.03.20
 */
#define MIN_WAIT_TIME 2
#define MAX_WAIT_TIME 50
#define STEP_WAIT_TIME 10

image get_image_from_stream(CvCapture *cap);

static char **camera_names;
static image **camera_alphabet;
static int camera_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
#ifdef SUPPORT_SECOND_CAMERA
static float **probs_in;
static box *boxes_in;
static image in_in   ;
static image in_s_in ;
static image det_in  ;
static image det_s_in;
static image disp_in = {0};
static CvCapture *cap_in; // Indoor Camera
static float *predictions_in[FRAMES];
static int camera_index_in = 0;
static image images_in[FRAMES];
static float *avg_in;
#endif

static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float camera_thresh = 0;
static float *predictions[FRAMES];
static int camera_index = 0;
static image images[FRAMES];
static float *avg;

#ifdef REDIS
redisContext    *gRedisContext = NULL;
#endif

int gFrameNum = 0;
int gFirstDetectDelayTime = 5;
int gSecondDetectDelayTime = 5;

#ifdef SUPPORT_SECOND_CAMERA
void *fetch_second_camera_in_thread(void *ptr)
{
    in_in = get_image_from_stream(cap_in);

    if(!in_in.data)
    {
        error("Stream closed.");
    }

    in_s_in = resize_image(in_in, net.w, net.h);

    //printf("image w: %d, h: %d\n", net.w, net.h);

    return 0;
}

void *detect_second_camera_in_thread(void *ptr)
{
    float nms = .4;

    layer l;
    float *X;
    float *prediction;
    int num_bbox = 0;
    int num_object = 0;
    int find_object = 0;

    l = net.layers[net.n-1];
    X = det_s_in.data;
    prediction = network_predict(net, X);

    memcpy(predictions_in[camera_index_in], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions_in, FRAMES, l.outputs, avg_in);
    l.output = avg_in;

    free_image(det_s_in);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, camera_thresh, probs_in, boxes_in, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, camera_thresh, probs_in, boxes_in, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }

    //if (nms > 0) do_nms(boxes_in, probs_in, l.w*l.h*l.n, l.classes, nms);
    if (nms > 0)
    {
        find_object = do_nms_with_threshold(boxes_in, probs_in, l.w*l.h*l.n, l.classes, nms, camera_thresh);
    }

    num_bbox = l.w*l.h*l.n;

    images_in[camera_index_in] = det_in;
    det_in = images_in[(camera_index_in + FRAMES/2 + 1)%FRAMES];
    camera_index_in = (camera_index_in + 1)%FRAMES;

    //printf("num bbox: %d\n", num_bbox);

    if (find_object == 0)
    {
        if (gSecondDetectDelayTime < MAX_WAIT_TIME)
        {
            gSecondDetectDelayTime += STEP_WAIT_TIME;
        }
        else gSecondDetectDelayTime = MAX_WAIT_TIME;

        return 0;
    }

    num_object = draw_detections(det_in, num_bbox, camera_thresh, boxes_in, probs_in, camera_names, camera_alphabet, 1);

    if (num_object > 0) gSecondDetectDelayTime = MIN_WAIT_TIME;
    else
    {
        if (gSecondDetectDelayTime < MAX_WAIT_TIME)
        {
            gSecondDetectDelayTime += STEP_WAIT_TIME;
        }
        else gSecondDetectDelayTime = MAX_WAIT_TIME;
    }

    return 0;
}
#endif

void *fetch_in_thread(void *ptr)
{
    in = get_image_from_stream(cap);

    if(!in.data)
    {
        error("Stream closed.");
    }

    in_s = resize_image(in, net.w, net.h);

    //printf("image w: %d, h: %d\n", net.w, net.h);

    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l;
    float *X;
    float *prediction;
    int num_bbox = 0;
    int num_object = 0;
    int find_object = 0;

    l = net.layers[net.n-1];
    X = det_s.data;
    prediction = network_predict(net, X);

    memcpy(predictions[camera_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, camera_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, camera_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }

    //if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    if (nms > 0)
    {
        find_object = do_nms_with_threshold(boxes, probs, l.w*l.h*l.n, l.classes, nms, camera_thresh);
    }

    num_bbox = l.w*l.h*l.n;

    images[camera_index] = det;
    det = images[(camera_index + FRAMES/2 + 1)%FRAMES];
    camera_index = (camera_index + 1)%FRAMES;

    /*
     * Added by Seongho Baek 2019.03.20
     * By version up the captured frame, old bbox would be invalidated.
     */
    version_up();

    //printf("num bbox: %d\n", num_bbox);
    if (find_object == 0)
    {
        if (gFirstDetectDelayTime < MAX_WAIT_TIME)
        {
            gFirstDetectDelayTime += STEP_WAIT_TIME;
        }
        else gFirstDetectDelayTime = MAX_WAIT_TIME;

        //invalidate(); // Invalidate all labels in memory. Added by Seongho Baek 2019.03.20

        return 0;
    }

#ifdef REDIS
    num_object = draw_and_send_detections(gRedisContext, det, num_bbox, camera_thresh, boxes, probs, camera_names, camera_alphabet, gFrameNum);

    if (num_object > 0) gFirstDetectDelayTime = MIN_WAIT_TIME;
    else
    {
        if (gFirstDetectDelayTime < MAX_WAIT_TIME)
        {
            gFirstDetectDelayTime += STEP_WAIT_TIME;
        }
        else gFirstDetectDelayTime = MAX_WAIT_TIME;
    }
#else
    draw_detections(det, num_bbox, camera_thresh, boxes, probs, camera_names, camera_alphabet, camera_classes);
#endif

    return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


image shot_to_save;
char* shot_filename_to_save = 0;
int   shot_seq = 0;

/* Thread for saving captured image. Added by Seongho Baek 2019.03.18 */
void *save_shot_in_thread(void *ptr)
{
    if (shot_filename_to_save)
    {
        char buff[80];

        shot_seq++;
        shot_seq %=2;

        //save_image(shot_to_save, shot_filename_to_save);
        memset(buff, 0, sizeof(buff));
        sprintf(buff, "../robot/node/public/screenshot%d\0", shot_seq);

        save_image(shot_to_save, buff);

        if (shot_seq == 0)
        {
            /* Mitigate flickering . Added by Seongho Baek 2019.03.18 */
            memset(buff, 0, sizeof(buff));
            sprintf(buff, "../robot/node/public/screenshot%d.jpg\0", shot_seq);
            rename("../robot/node/public/screenshot0.jpg", "../robot/node/public/screenshot.jpg");
        }

        shot_filename_to_save = 0;
    }

    return 0;
}

void run_camera(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *local_server, const char* redis_server, char **names, int classes)
{
    image **alphabet = load_alphabet();
    int delay = 1;

    const char *hostname = redis_server;
    int port = 6379;
    struct timeval timeout = {1, 500000};

    camera_names = names;
    camera_alphabet = alphabet;
    camera_classes = classes;
    camera_thresh = thresh;

    printf("Parse network config: %s\n", cfgfile);

    net = parse_network_cfg(cfgfile);

    if (weightfile) {
        printf("Load weight file\n");

        load_weights(&net, weightfile);
    }

    printf("Set bactch 1\n");

    set_batch_network(&net, 1);

    srand(2222222);

    /*
    if (filename) {
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    } else {

    }
     */

    printf("Start Camera Capture \n");

#ifdef SUPPORT_SECOND_CAMERA
    cap = cvCaptureFromCAM(0);
    cap_in = cvCaptureFromCAM(1);
#else
    cap = cvCaptureFromCAM(0);
#endif

    if (!cap) {
        error("Couldn't connect to webcam.\n");

        return;
    }

#ifdef SUPPORT_SECOND_CAMERA
    if (!cap_in) {
        error("Couldn't connect to webcam.\n");

        return;
    }
#endif

#ifdef REDIS
    gRedisContext = redisConnectWithTimeout(hostname, port, timeout);

    if (gRedisContext == NULL || gRedisContext->err) {
        if (gRedisContext) {
            printf("Connection Error: %s\n", gRedisContext->errstr);
            redisFree(gRedisContext);
            gRedisContext = NULL;
        } else {
            printf("Connection error: can't allocate redis context\n");
        }
    }
#endif

    run_identifier(local_server);

    layer l = net.layers[net.n - 1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for (j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for (j = 0; j < FRAMES; ++j) images[j] = make_image(1, 1, 3);

    boxes = (box *) calloc(l.w * l.h * l.n, sizeof(box));
    probs = (float **) calloc(l.w * l.h * l.n, sizeof(float *));
    for (j = 0; j < l.w * l.h * l.n; ++j) probs[j] = (float *) calloc(l.classes, sizeof(float *));

#ifdef SUPPORT_SECOND_CAMERA
    avg_in = (float *) calloc(l.outputs, sizeof(float));
    for (j = 0; j < FRAMES; ++j) predictions_in[j] = (float *) calloc(l.outputs, sizeof(float));
    for (j = 0; j < FRAMES; ++j) images_in[j] = make_image(1, 1, 3);

    boxes_in = (box *) calloc(l.w * l.h * l.n, sizeof(box));
    probs_in = (float **) calloc(l.w * l.h * l.n, sizeof(float *));
    for (j = 0; j < l.w * l.h * l.n; ++j) probs_in[j] = (float *) calloc(l.classes, sizeof(float *));
#endif

    pthread_t fetch_thread;
    pthread_t detect_thread;
    pthread_t fetch_second_thread;
    pthread_t detect_second_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    for (j = 0; j < FRAMES / 2; ++j) {
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

    for (j = 0; j < FRAMES / 2; ++j) {
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

#ifdef SUPPORT_SECOND_CAMERA
    fetch_second_camera_in_thread(0);
    det_in = in_in;
    det_s_in = in_s_in;

    fetch_second_camera_in_thread(0);
    detect_second_camera_in_thread(0);
    disp_in = det;
    det_in = in_in;
    det_s_in = in_s_in;

    for (j = 0; j < FRAMES / 2; ++j) {
        fetch_second_camera_in_thread(0);
        detect_second_camera_in_thread(0);
        disp_in = det_in;
        det_in = in_in;
        det_s_in = in_s_in;
    }

    for (j = 0; j < FRAMES / 2; ++j) {
        fetch_second_camera_in_thread(0);
        detect_second_camera_in_thread(0);
        disp_in = det_in;
        det_in = in_in;
        det_s_in = in_s_in;
    }
#endif

#ifdef SHOW_WINDOW
    cvNamedWindow("Camera", CV_WINDOW_NORMAL);
    cvMoveWindow("Camera", 0, 0);
    //cvResizeWindow("Camera", 1200, 900); // 1352 , 1013
    cvResizeWindow("Camera", 1920, 1080); // 1352 , 1013
#endif

    while (1) {
        pthread_t image_save_thread_t = -1;

        gFrameNum++;
        gFrameNum %= 100;

        if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

        if (gFrameNum % 2 == 0)
        {
            //save_image(disp, "../robot/node/public/screenshot");

            shot_filename_to_save = "../robot/node/public/screenshot";
            shot_to_save = disp;

			/* Thread for saving captured image. Added by Seongho Baek 2019.03.18 */
            pthread_create(&image_save_thread_t, 0, save_shot_in_thread, 0);
        }

#ifdef SHOW_WINDOW
        show_image(disp, "Camera");

        int c = cvWaitKey(delay);
#endif

        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);

        if (image_save_thread_t != -1)
            pthread_join(image_save_thread_t, 0);

#ifdef SHOW_WINDOW
        if (c == 10)
        {
            delay += 10;

            if (delay > 40) delay = 1;
        }
#endif
        free_image(disp);
        disp = det;
        det = in;
        det_s = in_s;

#ifdef SUPPORT_SECOND_CAMERA

        // Second Camera: Don't need to show display window.
        if (pthread_create(&fetch_second_thread, 0, fetch_second_camera_in_thread, 0)) error("Thread creation failed");
        if (pthread_create(&detect_second_thread, 0, detect_second_camera_in_thread, 0)) error("Thread creation failed");

        pthread_join(fetch_second_thread, 0);
        pthread_join(detect_second_thread, 0);

        free_image(disp_in);
        disp_in = det_in;

        det_in = in_in;
        det_s_in = in_s_in;
#endif

        if (gFirstDetectDelayTime <= gSecondDetectDelayTime)
        {
            cvWaitKey(gFirstDetectDelayTime);
        }
        else
        {
            cvWaitKey(gSecondDetectDelayTime);
        }
    }

    cvDestroyWindow("Camera");
    cvReleaseCapture(cap);

#ifdef REDIS
    if (gRedisContext) {
        redisFree(gRedisContext);
        gRedisContext = NULL;
    }
#endif
}

void detect_camera(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
    image **alphabet = load_alphabet();
    int delay = frame_skip;

    const char      *hostname = "127.0.0.1";
    int             port = 6379;
    struct timeval  timeout = {1, 500000};

    camera_names = names;
    camera_alphabet = alphabet;
    camera_classes = classes;
    camera_thresh = thresh;

    printf("Parse network config: %s\n", cfgfile);

    net = parse_network_cfg(cfgfile);

    if(weightfile)
    {
        printf("Load weight file\n");

        load_weights(&net, weightfile);
    }

    printf("Set bactch 1\n");

    set_batch_network(&net, 1);

    srand(2222222);

    if(filename)
    {
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }
    else
    {
        printf("Start Camera Capture \n");

        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap)
    {
        error("Couldn't connect to webcam.\n");

        return;
    }

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j)
    {
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix)
    {
        cvNamedWindow("Camera", CV_WINDOW_NORMAL);
        cvMoveWindow("Camera", 0, 0);
        //cvResizeWindow("Camera", 1352, 1013);
        cvResizeWindow("Camera", 1024, 768);
    }

    double before = get_wall_time();

    while(1)
    {
        ++count;
        gFrameNum++;
        gFrameNum %= 100;

        if(1)
        {
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            /*
            if(!prefix)
            {
                show_image(disp, "Camera");

                int c = cvWaitKey(1);

                if (c == 10)
                {
                    pthread_join(fetch_thread, 0);
                    pthread_join(detect_thread, 0);

                    break;

                    if(frame_skip == 0) frame_skip = 60;
                    else if(frame_skip == 4) frame_skip = 0;
                    else if(frame_skip == 60) frame_skip = 4;
                    else frame_skip = 0;
                }
            }
            else
            {
                char buff[256];
                sprintf(buff, "%s_%08d", prefix, count);
                save_image(disp, buff);
            }
            */

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if(delay == 0)
            {
                free_image(disp);
                disp  = det;
            }

            det   = in;
            det_s = in_s;
        }
        else
        {
            fetch_in_thread(0);
            det   = in;
            det_s = in_s;
            detect_in_thread(0);
            if(delay == 0) {
                free_image(disp);
                disp = det;
            }
            show_image(disp, "Camera");
            cvWaitKey(1);
        }

        --delay;

        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }

        sleep(0);
    }

    cvDestroyWindow("Camera");
    cvReleaseCapture(cap);
}


