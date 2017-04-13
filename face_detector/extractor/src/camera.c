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
#endif

#include "identifier.h"

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

    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    //printf("\033[2J");
    //printf("\033[1;1H");
    //printf("\nFPS:%.1f\n",fps);
    //printf("\nFPS:%.1f\n",fps);
    //printf("Objects:\n\n");

    num_bbox = l.w*l.h*l.n;

    images[camera_index] = det;
    det = images[(camera_index + FRAMES/2 + 1)%FRAMES];
    camera_index = (camera_index + 1)%FRAMES;

    //printf("num bbox: %d\n", num_bbox);

#ifdef REDIS
    draw_and_send_detections(gRedisContext, det, num_bbox, camera_thresh, boxes, probs, camera_names, camera_alphabet, gFrameNum);

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

    cap = cvCaptureFromCAM(CV_CAP_ANY);

    if (!cap) {
        error("Couldn't connect to webcam.\n");

        return;
    }

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

    for (j = 0; j < FRAMES / 2; ++j) {
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

#ifdef SHOW_WINDOW
    cvNamedWindow("Camera", CV_WINDOW_NORMAL);
    cvMoveWindow("Camera", 0, 0);
    cvResizeWindow("Camera", 1200, 900); // 1352 , 1013
#endif

    while (1) {
        gFrameNum++;
        gFrameNum %= 100;

        if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

        if (gFrameNum % 30 == 0)
        {
            save_image(disp, "../robot/node/public/screenshot");
        }

#ifdef SHOW_WINDOW
        show_image(disp, "Camera");

        int c = cvWaitKey(delay);
#endif

        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);

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

        sleep(0);
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


