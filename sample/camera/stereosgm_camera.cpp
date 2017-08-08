/*
Copyright 2016 fixstars

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libsgm.h>

#include "demo.h"
#include "renderer.h"

int main(int argc, char* argv[])
{
    bool isToFlipRightImage = (argc >= 2) && strcmp(argv[1], "--flipRight");
    if (isToFlipRightImage) { std::cout << "Flip right camera image horizontally." << std::endl; }
    const int horizontalFlipCode = 1;

    cv::VideoCapture leftCapture = cv::VideoCapture(0);
    cv::VideoCapture rightCapture = cv::VideoCapture(1);
    cv::Mat leftColor, rightColor;
    cv::Mat rightColorFlipped;
    cv::Mat left, right;

    bool hr = leftCapture.read(leftColor);
    hr = hr && rightCapture.read(rightColor);
    if (hr == false) {
        std::cerr << "It needs 2 cameras." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (left.size() != right.size() || left.type() != right.type()) {
        std::cerr << "mismatch input image size" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cv::cvtColor(leftColor, left, CV_BGR2GRAY);
    if (isToFlipRightImage)
    {
        cv::flip(rightColor, rightColorFlipped, horizontalFlipCode);
        cv::cvtColor(rightColorFlipped, right, CV_BGR2GRAY);
    }
    else
    {
        cv::cvtColor(rightColor, right, CV_BGR2GRAY);
    }

    int bits = 0;
    switch (left.type()) {
    case CV_8UC1: bits = 8; break;
    case CV_16UC1: bits = 16; break;
    default:
        std::cerr << "invalid input image color format. Type: " << left.type() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int disp_size = 64;
    int width = left.cols;
    int height = left.rows;

    cudaGLSetGLDevice(0);

    SGMDemo demo(width, height);

    if (demo.init()) {
        printf("fail to init SGM Demo\n");
        std::exit(EXIT_FAILURE);
    }

    sgm::StereoSGM ssgm(width, height, disp_size, bits, 16, sgm::EXECUTE_INOUT_HOST2CUDA);

    Renderer renderer(width, height);

    uint16_t* d_output_buffer = NULL;

    int frame_no = 0;
    while (!demo.should_close())
    {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);


        bool hr = leftCapture.read(leftColor);
        hr = hr && rightCapture.read(rightColor);
        if (hr == false) {
            std::cerr << "Failed capturing." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        cv::cvtColor(leftColor, left, CV_BGR2GRAY);
        if (isToFlipRightImage)
        {
            cv::flip(rightColor, rightColorFlipped, horizontalFlipCode);
            cv::cvtColor(rightColorFlipped, right, CV_BGR2GRAY);
        }
        else
        {
            cv::cvtColor(rightColor, right, CV_BGR2GRAY);
        }

#if _DEBUG || 1
        cv::imshow("left", left);
        cv::imshow("right", right);
#endif

        ssgm.execute(left.data, right.data, (void**)&d_output_buffer); // , sgm::DST_TYPE_CUDA_PTR, 16);

        switch (demo.get_flag()) {
        case 0:
        {
            renderer.render_input((uint16_t*)left.data);
        }
        break;
        case 1:
            renderer.render_disparity(d_output_buffer, disp_size);
            break;
        case 2:
            renderer.render_disparity_color(d_output_buffer, disp_size);
            break;
        }

        demo.swap_buffer();
        frame_no++;
    }
}
