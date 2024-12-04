#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <iomanip>

class SlideExtractor {
private:
    cv::VideoCapture video;
    int start_frame, end_frame;
    double similarity_threshold = 0.9;
    cv::Rect slide_roi;

    // Detect if two frames represent different slides
    bool is_slide_changed(const cv::Mat& prev_frame, const cv::Mat& curr_frame) {
        cv::Mat diff;
        cv::absdiff(prev_frame, curr_frame, diff);
        cv::Mat gray_diff;
        cv::cvtColor(diff, gray_diff, cv::COLOR_BGR2GRAY);
        
        double similarity = 1.0 - (cv::countNonZero(gray_diff) / (double)(diff.rows * diff.cols));
        return similarity < similarity_threshold;
    }

    // Find largest white region as potential slide
    cv::Rect detect_slide_region(const cv::Mat& frame) {
        cv::Mat white_mask;
        cv::inRange(frame, cv::Scalar(200, 200, 200), cv::Scalar(255, 255, 255), white_mask);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(white_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) return cv::Rect(0, 0, frame.cols, frame.rows);
        
        auto largest_contour = *std::max_element(
            contours.begin(), contours.end(), 
            [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); }
        );
        
        return cv::boundingRect(largest_contour);
    }

public:
    SlideExtractor(const std::string& video_path, int start = 0, int end = -1) {
        video.open(video_path);
        if (!video.isOpened()) {
            throw std::runtime_error("Cannot open video file");
        }
        
        start_frame = start;
        end_frame = (end == -1) ? video.get(cv::CAP_PROP_FRAME_COUNT) : end;
    }

    void extract_slides(const std::string& output_dir) {
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        cv::Mat prev_frame, curr_frame;
        int slide_count = 0;
        int total_frames = 0;

        video.set(cv::CAP_PROP_POS_FRAMES, start_frame);

        while (total_frames < end_frame - start_frame) {
            video >> curr_frame;
            if (curr_frame.empty()) break;

            // First frame or slide change detection
            if (prev_frame.empty() || is_slide_changed(prev_frame, curr_frame)) {
                slide_roi = detect_slide_region(curr_frame);
                
                // Crop slide region
                cv::Mat slide = curr_frame(slide_roi);
                
                // Save slide image
                std::ostringstream filename;
                filename << output_dir << "/result_slide" 
                         << std::setw(2) << std::setfill('0') << slide_count 
                         << "_frame" 
                         << std::setw(4) << std::setfill('0') << total_frames 
                         << ".png";
                
                cv::imwrite(filename.str(), slide);
                slide_count++;
            }

            prev_frame = curr_frame.clone();
            total_frames++;
        }

        std::cout << "Extracted " << slide_count << " slides." << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <video_path> <output_directory> [start_frame] [end_frame]" 
                  << std::endl;
        return 1;
    }

    try {
        std::string video_path = argv[1];
        std::string output_dir = argv[2];
        
        int start_frame = (argc > 3) ? std::stoi(argv[3]) : 0;
        int end_frame = (argc > 4) ? std::stoi(argv[4]) : -1;

        SlideExtractor extractor(video_path, start_frame, end_frame);
        extractor.extract_slides(output_dir);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}