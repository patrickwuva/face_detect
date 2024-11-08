#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

bool is_new_face(const cv::Rect& face, const cv::Rect& last_face, double position_threshold, double size_threshold) {
    // Calculate the center of each face
    cv::Point center_face(face.x + face.width / 2, face.y + face.height / 2);
    cv::Point center_last_face(last_face.x + last_face.width / 2, last_face.y + last_face.height / 2);

    // Calculate distance between centers and size difference
    double distance = cv::norm(center_face - center_last_face);
    double size_difference = std::abs(face.area() - last_face.area()) / (double)last_face.area();

    // Check if the distance or size difference is above the thresholds
    return (distance > position_threshold || size_difference > size_threshold);
}

int main(int argc, char **argv) {
    // Load the pre-trained face detection model
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("../haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade\n";
        return -1;
    }

    // Open the default camera
    cv::VideoCapture video_capture(0);
    if (!video_capture.isOpened()) {
        std::cerr << "Error opening video stream\n";
        return -1;
    }

    // Set camera resolution for better detection (adjust to your camera's capabilities)
    video_capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    int face_count = 0;  // Counter for naming saved face images
    int frame_count = 0; // Frame counter to control save frequency
    int save_interval = 30; // Save an image every 30 frames
    cv::Rect last_saved_face; // Store the last saved face's position and size

    // Define thresholds for determining a "new" face
    double position_threshold = 100.0; // Adjust based on camera distance and movement
    double size_threshold = 0.3; // 30% size change required to consider as a new face

    cv::Mat frame;
    while (true) {
        video_capture >> frame;
        if (frame.empty()) break;

        // Convert to grayscale and preprocess for better face detection
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray_frame, gray_frame, cv::Size(3, 3), 0); // Reduce noise
        cv::equalizeHist(gray_frame, gray_frame); // Improve contrast

        // Detect faces with further increased minimum face size
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(
            gray_frame, faces, 
            1.1,
            5,
            0,
            cv::Size(200, 200),
            cv::Size(600, 600)
        );

        for (size_t i = 0; i < faces.size(); i++) {
            if (frame_count % save_interval == 0 && (last_saved_face.empty() || is_new_face(faces[i], last_saved_face, position_threshold, size_threshold))) {
                cv::Mat face = frame(faces[i]); // Extract the face region

                std::string filename = "face_" + std::to_string(face_count++) + ".jpg";
                cv::imwrite(filename, face);

                last_saved_face = faces[i];
            }

            cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Face Detection", frame);

        if (cv::waitKey(10) == 27) break;

        frame_count++;
    }

    video_capture.release();
    cv::destroyAllWindows();

    return 0;
}
