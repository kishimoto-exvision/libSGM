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

#include <nppi.h>
#include <GL/glew.h>
#include <libsgm.h>
#include "demo.h"
#include "renderer.h"

#include <sl/Camera.hpp>


class DepthEstimationFromZedCameraBase
{
public:
    sl::InitParameters initParameters;
    sl::RuntimeParameters runtimeParameters;
    sl::Camera zed;
    sl::ERROR_CODE err;

    bool isToQuit;
    sl::Mat left_zm;
    sl::Mat right_zm;

public:
    virtual bool Initialize() = 0;
    virtual bool ProcessOneFrame() = 0;
    virtual void Close() = 0;
};



class DepthEstimationByZedSdk : public DepthEstimationFromZedCameraBase
{
public:
    sl::Mat depth_zm;

public:
    DepthEstimationByZedSdk()
    {
        isToQuit = false;

        initParameters.coordinate_units = sl::UNIT_METER;
        initParameters.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;

        //initParameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
        initParameters.depth_mode = sl::DEPTH_MODE_QUALITY;
        //runtimeParameters.sensing_mode = sl::SENSING_MODE_STANDARD;
        runtimeParameters.sensing_mode = sl::SENSING_MODE_FILL;
        runtimeParameters.enable_depth = true;
        runtimeParameters.enable_point_cloud = true;
        runtimeParameters.move_point_cloud_to_world_frame = true;

        if (false)
        {
            initParameters.camera_resolution = sl::RESOLUTION::RESOLUTION_HD720;
            initParameters.camera_fps = 60;
        }
        else
        {
            initParameters.svo_input_filename.set("C:\\Users\\KISHIMOTO\\Documents\\ZED\\HD720_SN13619_10-42-37.svo");
            //initParameters.svo_real_time_mode = true;
        }
    }

    bool Initialize()
    {
        err = zed.open(initParameters);
        if (err != sl::SUCCESS) { return false; }
        return true;
    }

    bool ProcessOneFrame()
    {
        err = zed.grab(runtimeParameters);
        if (err == sl::ERROR_CODE_NOT_A_NEW_FRAME) { return false; }
        err = zed.retrieveImage(left_zm, sl::VIEW_LEFT, sl::MEM_CPU);
        err = zed.retrieveImage(right_zm, sl::VIEW_RIGHT, sl::MEM_CPU);
        err = zed.retrieveImage(depth_zm, sl::VIEW_DEPTH, sl::MEM_CPU);
        cv::imshow("Left", cv::Mat(left_zm.getHeight(), left_zm.getWidth(), CV_8UC4, left_zm.getPtr<sl::uchar1>(sl::MEM_CPU)));
        cv::imshow("Right", cv::Mat(right_zm.getHeight(), right_zm.getWidth(), CV_8UC4, right_zm.getPtr<sl::uchar1>(sl::MEM_CPU)));
        // NOTE: runtimeParameters.enable_depth must be true!
        cv::Mat depthImage = cv::Mat(depth_zm.getHeight(), depth_zm.getWidth(), CV_8UC4, depth_zm.getPtr<sl::uchar1>(sl::MEM_CPU));
        cv::imshow("Depth", depthImage);
        cv::waitKey(1);
        return true;
    }

    void Close()
    {
        isToQuit = true;
        zed.close();
        left_zm.free(sl::MEM_CPU);
        right_zm.free(sl::MEM_CPU);
        depth_zm.free(sl::MEM_CPU);
    }
};



class DepthEstimationByLibSgm : public DepthEstimationFromZedCameraBase
{
public:
    int disp_size;

public:
    NppiSize roi;
    sgm::StereoSGM* ssgm;
    SGMDemo* demo;
    Renderer* renderer;
    cv::Mat* h_input_left;
    uint16_t* d_output_buffer;
    uint8_t* d_input_left;
    uint8_t* d_input_right;

public:
    DepthEstimationByLibSgm()
    {
        isToQuit = false;

        disp_size = 64;
        ssgm = nullptr;
        demo = nullptr;
        renderer = nullptr;

        h_input_left = nullptr;
        d_output_buffer = nullptr;
        d_input_left = nullptr;
        d_input_right = nullptr;


        initParameters.coordinate_units = sl::UNIT_METER;
        initParameters.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;

        //initParameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
        initParameters.depth_mode = sl::DEPTH_MODE_QUALITY;
        //runtimeParameters.sensing_mode = sl::SENSING_MODE_STANDARD;
        runtimeParameters.sensing_mode = sl::SENSING_MODE_FILL;
        runtimeParameters.enable_depth = false;
        runtimeParameters.enable_point_cloud = false;
        runtimeParameters.move_point_cloud_to_world_frame = false;

        if (false)
        {
            initParameters.camera_resolution = sl::RESOLUTION::RESOLUTION_HD720;
            initParameters.camera_fps = 60;
        }
        else
        {
            initParameters.svo_input_filename.set("C:\\Users\\KISHIMOTO\\Documents\\ZED\\HD720_SN13619_10-42-37.svo");
            //initParameters.svo_real_time_mode = true;
        }
    }

    bool Initialize()
    {
        err = zed.open(initParameters);
        if (err != sl::SUCCESS) { return false; }
        roi.width = zed.getResolution().width;
        roi.height = zed.getResolution().height;
        ssgm = new sgm::StereoSGM(roi.width, roi.height, disp_size, 8, 16, sgm::EXECUTE_INOUT_CUDA2CUDA);
        demo = new SGMDemo(roi.width, roi.height);
        if (demo->init())
        {
            printf("fail to init SGM Demo\n");
            Close();
            std::exit(EXIT_FAILURE);
        }
        renderer = new Renderer(roi.width, roi.height);
        cudaMalloc((void**)&d_input_left, roi.width *roi.height);
        cudaMalloc((void**)&d_input_right, roi.width *roi.height);
        h_input_left = new cv::Mat(roi.height, roi.width, CV_8UC1);
        return true;
    }

    bool ProcessOneFrame()
    {
        if (demo->should_close()) { return false; }

        err = zed.grab(runtimeParameters);
        if (err == sl::ERROR_CODE_NOT_A_NEW_FRAME) { return false; }
        err = zed.retrieveImage(left_zm, sl::VIEW_LEFT, sl::MEM_GPU);
        err = zed.retrieveImage(right_zm, sl::VIEW_RIGHT, sl::MEM_GPU);
        nppiRGBToGray_8u_AC4C1R((const Npp8u*)left_zm.getPtr<sl::uchar1>(sl::MEM_GPU), roi.width * 4, d_input_left, roi.width, roi);
        nppiRGBToGray_8u_AC4C1R((const Npp8u*)right_zm.getPtr<sl::uchar1>(sl::MEM_GPU), roi.width * 4, d_input_right, roi.width, roi);

        ssgm->execute(d_input_left, d_input_right, (void**)&d_output_buffer);

        switch (demo->get_flag())
        {
        case 0:
            cudaMemcpy(h_input_left->data, d_input_left, roi.width * roi.height, cudaMemcpyDeviceToHost);
            renderer->render_input((uint8_t*)h_input_left->data);
            break;
        case 1:
            renderer->render_disparity(d_output_buffer, disp_size);
            break;
        case 2:
            renderer->render_disparity_color(d_output_buffer, disp_size);
            break;
        }

        demo->swap_buffer();
        return true;
    }

    void Close()
    {
        isToQuit = true;
        zed.close();
        if (ssgm != nullptr) { delete ssgm; ssgm = nullptr; }
        if (demo != nullptr) { delete demo; demo = nullptr; }
        if (renderer != nullptr) { delete renderer; renderer = nullptr; }
        if (h_input_left != nullptr) { delete h_input_left; h_input_left = nullptr; }
        left_zm.free(sl::MEM_GPU);
        right_zm.free(sl::MEM_GPU);
        cudaFree(d_input_left);
        cudaFree(d_input_right);
    }
};



int main(int argc, char* argv[])
{
    DepthEstimationFromZedCameraBase* depthEstimation;
    depthEstimation = new DepthEstimationByZedSdk();
    //depthEstimation = new DepthEstimationByLibSgm();

    if (depthEstimation->Initialize() == false)
    {
        std::cout << sl::errorCode2str(depthEstimation->err) << std::endl;
        exit(EXIT_FAILURE);
    }
    while (!depthEstimation->isToQuit)
    {
        if (depthEstimation->ProcessOneFrame() == false) { depthEstimation->isToQuit = true; break; }
    }
    depthEstimation->Close();

    return 0;
}
