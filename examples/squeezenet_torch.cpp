// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <ctime>
#include <iostream>

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN


static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    squeezenet.load_param("squeezenet-torch.param");
    squeezenet.load_model("squeezenet-torch.bin");

    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, 224, 224);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(0, 0);

    clock_t begin = clock();
    double elapsed_secs = 0;
    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("0", in);
    clock_t insert_inputs = clock();
    elapsed_secs = static_cast<double>(insert_inputs - begin) / CLOCKS_PER_SEC;
    std::cout << "insert_inputs :" << elapsed_secs << " s" << std::endl;

    ncnn::Mat out;
    ex.extract("127", out);

    clock_t extract_out = clock();
    elapsed_secs = static_cast<double>(extract_out - begin) / CLOCKS_PER_SEC;
    std::cout << "extract_out :" << elapsed_secs << " s" << std::endl;

    {
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        softmax->load_param(pd);

        softmax->forward_inplace(out);

        delete softmax;
    }
    cls_scores.resize(out.w * out.h * out.c);

    for (int j=0; j<10; j++)
    {
        fprintf(stderr, "cv::imread %.3f \n", out[j]);
    }
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }
    clock_t end = clock();
    elapsed_secs = static_cast<double>(end - begin) / CLOCKS_PER_SEC;
    std::cout << "end :" << elapsed_secs << " s" << std::endl;



    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    cv::resize(m, m, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    print_topk(cls_scores, 3);

    return 0;
}
