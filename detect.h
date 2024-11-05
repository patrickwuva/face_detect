#ifndef DETECT_H
#define DETECT_H
#include <opencv2/dnn.hpp>

class detect {
public:
    explicit detect();

    std::vector<cv::Rect> detect_face_rectangles(const cv::Mat &frame);

private:
    cv::dnn::Net network_;
    const int input_image_width_;
    const int input_image_height_;
    const double scale_factor_;
    const cv::Scalar mean_values_;
    const float confidence_theshold_;
};

#endif
