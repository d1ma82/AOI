#include<iso646.h>
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/core/utility.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include"utils/multiple_window.h"

using namespace std;
using namespace cv;

shared_ptr<MultipleImageWindow> miw;

const char* keys = {

    "{help h usage ? || print this message}"
    "{@image || Input image}"
    "{@lightPattern || Image light pattern to apply to image input}"
    "{lightMethod |1| Method to remove background light, 0 difference, 1 divide, 2 no light remove}"
    "{segMethod |1| Method to segment: 1 connected Components, 2 connected Components with stats, 3 find Contours}"
};

void mat_info (Mat mat) {

    cout << "size:    \t" << mat.cols << 'x' << mat.rows << endl
         << "channels:\t" << mat.channels() << endl 
         << "depth:   \t" << mat.depth() << endl;
}

Mat do_remove_light (Mat img, Mat pattern, int method) {

    Mat aux;
    if (method == 1) {

        // Require change our image to 32bit float for division
        Mat img32, pattern32;
        img.convertTo(img32, CV_32F);
        pattern.convertTo(pattern32, CV_32F);
        //Divide image by the pattern
        aux = 1 - (img32 / pattern32);
        // Convert to 8 bit and scale
        aux.convertTo(aux, CV_8U, 255);
    } else {
        aux = pattern - img;
    }
    return aux;
}

Mat calc_light_pattern(Mat img) {

    Mat pattern;
    // Basic and effective way to calculate light pattern from image
    blur(img, pattern, Size(img.cols/3, img.rows/3));
    return pattern;
}

static Scalar randomColor(RNG& rng) {

	auto icolor = (unsigned) rng;
	return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

Mat connected_components(Mat input) {

    // Use connected comp to divide our image in multiple connected comp objects
    Mat labels;
    auto num_objects = connectedComponents(input, labels, 4);
    cout << "Total objects: " << num_objects << endl;
    if (num_objects < 2) return input;

    //Create output image coloring the objects
    Mat output = Mat::zeros(input.rows, input.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    for (auto i=1; i<num_objects; i++) {

        Mat mask = labels == i;
        output.setTo(randomColor(rng), mask);
    }
    return output;
}

Mat connected_components_stats(Mat input) {

    Mat labels, stats, centroids;
    auto num_objects = connectedComponentsWithStats(input, labels, stats, centroids);
    cout << "Total objects: " << num_objects << endl;
    //Check the number of objects detected
    if (num_objects < 2) return input;

    //Create output image coloring the objects and show area
    Mat output = Mat::zeros(input.rows, input.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    for (auto i=1; i<num_objects; i++) {

        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 1000) continue;
        cout << "Object " << i << " with pos: " << centroids.at<Point2d>(i) 
             << " with area " << area << endl;
        Mat mask = labels == i;
        output.setTo(randomColor(rng), mask);
        // draw text with area
        stringstream ss;
        ss << "area: " << stats.at<int>(i, CC_STAT_AREA);

        putText(output, 
                ss.str(), 
                centroids.at<Point2d>(i), 
                FONT_HERSHEY_SIMPLEX, 0.4, 
                Scalar(255, 255, 255));              
    }
       return output;
}

Mat find_contours_basic(Mat input) {

    vector<vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat output = Mat::zeros(input.rows, input.cols, CV_8UC3);
    // Check the number of objects detected
    if (contours.size() == 0) {
        cout << "No object detected\n";
        return input;
    }
    cout << "Objects detected: " << contours.size() << endl;
    RNG rng(0xFFFFFFFF);
    for (auto i=0; i<contours.size(); i++) {

        drawContours(output, contours, i, randomColor(rng));
    }
    return output;
}

int main(int argc, const char* argv[]) {

    CommandLineParser parser(argc, argv, keys);
    parser.about("Auto Object Inspection example.");
    if (parser.has("help")) {

        parser.printMessage();
        return 0;
    }

    string img_file = parser.get<string>(0);
    string light_pattern_file = parser.get<string>(1);
    auto method_light = parser.get<int>("lightMethod");
    auto method_seg = parser.get<int>("segMethod");

    if (not parser.check()) {

        parser.printMessage();
        return -1;
    }
    // Input image gray
    Mat img = imread(img_file, 0);
    
    if (img.data == NULL) {

        cout << "Error loading image " << img_file << endl;
        return -1;
    }
    mat_info(img);

    //Create multiple image window
    miw = make_shared<MultipleImageWindow>("Main window", 3, 2, WINDOW_AUTOSIZE);
    
    // Remove noise salt paper
    Mat img_noise;
    medianBlur(img, img_noise, 3);

    Mat light_pattern_img = imread(light_pattern_file, 0);
    if (light_pattern_img.data == NULL) {
        cout << "Calculate pattern\n";
        light_pattern_img = calc_light_pattern(img_noise);
    } else {
        if (light_pattern_img.channels() > 1) 
                cvtColor(light_pattern_img, light_pattern_img, COLOR_BGR2GRAY); 
    }
    medianBlur(light_pattern_img, light_pattern_img, 3);

    // Apply light pattern
    Mat img_no_light;
    img_noise.copyTo(img_no_light);
    if (method_light != 2)  
        img_no_light = do_remove_light(img_noise, light_pattern_img, method_light);
    
    // Binarize imgage
    Mat img_thr;
    if (method_light != 2) threshold(img_no_light, img_thr, 30, 255, THRESH_BINARY);
    else (threshold(img_noise, img_thr, 140, 255, THRESH_BINARY_INV));


    // show images
    miw->addImage("Input", img);
    miw->addImage("Input without noise", img_noise);
    miw->addImage("Light Pattern", light_pattern_img);
    miw->addImage("No light", img_no_light);
    miw->addImage("Threshold", img_thr);

    switch (method_seg) {

        case 1: miw->addImage("Connected Comp", connected_components(img_thr)); break;
        case 2: miw->addImage("Connected Comp Stats", connected_components_stats(img_thr)); break;
        case 3: miw->addImage("Contours", find_contours_basic(img_thr)); break;
    }
    miw->render();
    waitKey(0);
    return 0;
}
