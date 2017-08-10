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

#include <nppi.h>
#include <GL/glew.h>
#include <libsgm.h>
#include "demo.h"
#include "renderer.h"

#include <sl/Camera.hpp>

#if _DEBUG
#define OpenCVLibPath "../../../OpenCVBuiltFilesForWindows/opencv_x64/lib/Debug/"
#define OpenCVLibExt "d.lib"
#else
#define OpenCVLibPath "../../../OpenCVBuiltFilesForWindows/opencv_x64/lib/Release/"
#define OpenCVLibExt ".lib"
#endif
#pragma comment(lib, OpenCVLibPath "opencv_core320" OpenCVLibExt)
#pragma comment(lib, OpenCVLibPath "opencv_highgui320" OpenCVLibExt)
#pragma comment(lib, OpenCVLibPath "opencv_imgproc320" OpenCVLibExt)
#pragma comment(lib, OpenCVLibPath "opencv_calib3d320" OpenCVLibExt)



class DepthEstimationFromZedCameraBase
{
public:
    std::string svoVideoFilePath;
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
    }

    bool Initialize()
    {
        if (svoVideoFilePath.empty())
        {
            initParameters.camera_resolution = sl::RESOLUTION::RESOLUTION_HD720;
            initParameters.camera_fps = 60;
        }
        else
        {
            initParameters.svo_input_filename.set(svoVideoFilePath.c_str());
            //initParameters.svo_real_time_mode = true;
        }

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
    sl::MEM memType;
    sl::CameraInformation camInfo;
    cv::Mat map11, map12, map21, map22;

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
        memType = sl::MEM_CPU;


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
    }

    bool Initialize()
    {
        if (svoVideoFilePath.empty())
        {
            initParameters.camera_resolution = sl::RESOLUTION::RESOLUTION_HD720;
            initParameters.camera_fps = 60;
        }
        else
        {
            initParameters.svo_input_filename.set(svoVideoFilePath.c_str());
            //initParameters.svo_real_time_mode = true;
        }

        err = zed.open(initParameters);
        if (err != sl::SUCCESS) { return false; }
        roi.width = zed.getResolution().width;
        roi.height = zed.getResolution().height;

        camInfo = zed.getCameraInformation();
        auto&& calibParams = camInfo.calibration_parameters_raw;


        int i;
        // input
        cv::Mat M1(3, 3, CV_64FC1);
        cv::Mat M2(3, 3, CV_64FC1);
        cv::Mat D1(1, 5, CV_64FC1);
        cv::Mat D2(1, 5, CV_64FC1);
        cv::Mat R(3, 3, CV_64FC1);
        cv::Mat T(3, 1, CV_64FC1);
        cv::Size img_size = { roi.width, roi.height };
        // output
        cv::Mat R1, P1, R2, P2, Q;
        cv::Rect validRoi[2];

        M1.at<double>(0, 0) = calibParams.left_cam.fx;
        M1.at<double>(1, 1) = calibParams.left_cam.fy;
        M1.at<double>(0, 2) = calibParams.left_cam.cx;
        M1.at<double>(1, 2) = calibParams.left_cam.cy;
        M1.at<double>(2, 2) = 1.0;
        for (i = 0; i < 5; i++) { D1.at<double>(0, i) = calibParams.left_cam.disto[i]; }

        M2.at<double>(0, 0) = calibParams.right_cam.fx;
        M2.at<double>(1, 1) = calibParams.right_cam.fy;
        M2.at<double>(0, 2) = calibParams.right_cam.cx;
        M2.at<double>(1, 2) = calibParams.right_cam.cy;
        M2.at<double>(2, 2) = 1.0;
        for (i = 0; i < 5; i++) { D2.at<double>(0, i) = calibParams.right_cam.disto[i]; }

        //sl::float3 R; /*!< Rotation (using Rodrigues' transformation) between the two sensors. Defined as 'tilt', 'convergence' and 'roll'.*/
        cv::Mat tilt_convergence_roll(1, 3, CV_64FC1);
        for (i = 0; i < 3; i++) { tilt_convergence_roll.at<double>(0, i) = calibParams.R[i]; }
        cv::Rodrigues(tilt_convergence_roll, R);

        //sl::float3 T; /*!< Translation between the two sensors. T.x is the distance between the two cameras (baseline) in the sl::UNIT chosen during sl::Camera::open (mm, cm, meters, inches...).*/
        for (i = 0; i < 3; i++) { T.at<double>(i, 0) = calibParams.T[i]; }

        // exception occurs
        stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &validRoi[0], &validRoi[1]);
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);


        if (memType == sl::MEM_GPU)
        {
            ssgm = new sgm::StereoSGM(roi.width, roi.height, disp_size, 8, 16, sgm::EXECUTE_INOUT_CUDA2CUDA);
        }
        else if (memType == sl::MEM_CPU)
        {
            ssgm = new sgm::StereoSGM(roi.width, roi.height, disp_size, 8, 16, sgm::EXECUTE_INOUT_HOST2CUDA);
        }
        else
        {
            assert(false);
            return false;
        }

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

        if (memType == sl::MEM_GPU)
        {
            err = zed.retrieveImage(left_zm, sl::VIEW_LEFT, sl::MEM_GPU);
            err = zed.retrieveImage(right_zm, sl::VIEW_RIGHT, sl::MEM_GPU);
            nppiRGBToGray_8u_AC4C1R((const Npp8u*)left_zm.getPtr<sl::uchar1>(sl::MEM_GPU), roi.width * 4, d_input_left, roi.width, roi);
            nppiRGBToGray_8u_AC4C1R((const Npp8u*)right_zm.getPtr<sl::uchar1>(sl::MEM_GPU), roi.width * 4, d_input_right, roi.width, roi);
            ssgm->execute(d_input_left, d_input_right, (void**)&d_output_buffer);
        }
        else if (memType == sl::MEM_CPU)
        {
            err = zed.retrieveImage(left_zm, sl::VIEW_LEFT, sl::MEM_CPU);
            err = zed.retrieveImage(right_zm, sl::VIEW_RIGHT, sl::MEM_CPU);
            cv::Mat left_ocv = cv::Mat(left_zm.getHeight(), left_zm.getWidth(), CV_8UC4, left_zm.getPtr<sl::uchar1>(sl::MEM_CPU));
            cv::Mat right_ocv = cv::Mat(right_zm.getHeight(), right_zm.getWidth(), CV_8UC4, right_zm.getPtr<sl::uchar1>(sl::MEM_CPU));
            cv::Mat left_gray_ocv;
            cv::Mat right_gray_ocv;
            cv::cvtColor(left_ocv, left_gray_ocv, CV_BGRA2GRAY);
            cv::cvtColor(right_ocv, right_gray_ocv, CV_BGRA2GRAY);

            switch (1)
            {
            case 1:
            {
                if (false)
                {
                    cv::imshow("LeftGray", left_gray_ocv);
                    cv::imshow("RightGray", right_gray_ocv);
                    cv::waitKey(1);
                }
                ssgm->execute(left_gray_ocv.data, right_gray_ocv.data, (void**)&d_output_buffer);
                break;
            }
            case 2:
            {
                cv::Mat left_gray_bilateral_ocv;
                cv::Mat right_gray_bilateral_ocv;
                cv::bilateralFilter(left_gray_ocv, left_gray_bilateral_ocv, 7, 50, 150);
                cv::bilateralFilter(right_gray_ocv, right_gray_bilateral_ocv, 7, 50, 150);
                if (true)
                {
                    cv::imshow("LeftGray", left_gray_ocv);
                    cv::imshow("RightGray", right_gray_ocv);
                    cv::imshow("LeftGrayBilateral", left_gray_bilateral_ocv);
                    cv::imshow("RightGrayBilateral", right_gray_bilateral_ocv);
                    cv::waitKey(1);
                }
                ssgm->execute(left_gray_bilateral_ocv.data, right_gray_bilateral_ocv.data, (void**)&d_output_buffer);
                break;
            }
            case 3:
            {
                // The result is worse.
                // maybe the captured image with "zed.retrieveImage" is already undistorted.
                cv::Mat left_gray_rectify_ocv;
                cv::Mat right_gray_rectify_ocv;
                cv::remap(left_gray_ocv, left_gray_rectify_ocv, map11, map12, cv::INTER_LINEAR);
                cv::remap(right_gray_ocv, right_gray_rectify_ocv, map21, map22, cv::INTER_LINEAR);
                if (true)
                {
                    cv::imshow("LeftGrayRectify", left_gray_rectify_ocv);
                    cv::imshow("RightGrayRectify", right_gray_rectify_ocv);
                    cv::waitKey(1);
                }
                ssgm->execute(left_gray_rectify_ocv.data, right_gray_rectify_ocv.data, (void**)&d_output_buffer);
                break;
            }
            default:
                assert(false);
            }
        }
        else
        {
            assert(false);
            return false;
        }

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
        left_zm.free(memType);
        right_zm.free(memType);
        cudaFree(d_input_left);
        cudaFree(d_input_right);
    }
};



int main(int argc, char* argv[])
{
    DepthEstimationFromZedCameraBase* depthEstimation;
    //depthEstimation = new DepthEstimationByZedSdk();
    depthEstimation = new DepthEstimationByLibSgm();

    if (argc > 2) { depthEstimation->svoVideoFilePath = std::string(argv[1]); }
    else if (true) { depthEstimation->svoVideoFilePath = "C:\\Users\\KISHIMOTO\\Documents\\ZED\\HD720_SN13619_10-42-37.svo"; }
    else { depthEstimation->svoVideoFilePath = ""; } // use camera 

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
    delete depthEstimation;

    return 0;
}
