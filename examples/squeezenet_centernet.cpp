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

#include "squeezenet-centernet-sim.id.h"

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    squeezenet.load_param_bin("squeezenet-centernet-sim.param.bin");
    squeezenet.load_model("squeezenet-centernet-sim.bin");

    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 512, 512);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, 512, 512);

    const float mean_vals[3] = {0.408f, 0.447f, 0.470f};
    const float norm_vals[3] = {0.289f, 0.274f, 0.278f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.input(squeezenet_centernet_sim_param_id::BLOB_data, in);

    ncnn::Mat out;
    ex.extract(squeezenet_centernet_sim_param_id::BLOB_output, out);

    double elapsed_secs = 0;
    clock_t begin = clock();


    ex.input(squeezenet_centernet_sim_param_id::BLOB_data, in);
    ex.extract(squeezenet_centernet_sim_param_id::BLOB_output, out);

    clock_t extract_out = clock();
    elapsed_secs = static_cast<double>(extract_out - begin) / CLOCKS_PER_SEC;
    printf("%d %d %d\ncpu_time_used: %fs\n", out.w, out.h, out.c, elapsed_secs);
    printf("score: %f\n", out[69, 90, 0]);

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
    cv::resize(m, m, cv::Size(512, 512));
    m.convertTo(m, CV_32F, 1.0 / 255, 0);
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

    // print_topk(cls_scores, 3);

    return 0;
}
