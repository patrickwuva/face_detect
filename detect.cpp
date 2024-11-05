#include <sstream>
#include <vector>
#include <string>
#include <detect.h>
#include <opencv2/opencv.hpp>

detect::detect() :
    confidence_threshold_(0.5),
    input_image_height_(300),
    input_image_width_(300),
    scale_factor_(1.0),
    mean_values_({104.0, 177.0, 123.0}) {

        network_ = cv::dnn:readNetFromCaffe(FACE_DETECTION_CONFIGURATION, FACE_DETECTION_WEIGHTS);
    }
