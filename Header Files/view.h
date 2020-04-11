#ifndef VIEW_H
#define VIEW_H
#pragma once

#include "pch.h"

class ViewData {
public:
    cv::Mat imColor, imGray;
    cv::cuda::GpuMat d_imColor, d_imGray;
private:
    void setView(const cv::Mat imColor, const cv::Mat imGray) {
        imColor.copyTo(this->imColor);
        imGray.copyTo(this->imGray);
    }

    void setView(const cv::cuda::GpuMat d_imColor, const cv::cuda::GpuMat d_imGray) {
        d_imColor.copyTo(this->d_imColor);
        d_imGray.copyTo(this->d_imGray);
    }
public:
    ViewData() {}

    ViewData(const cv::Mat imColor, const cv::Mat imGray) {
        setView(imColor, imGray);
    }

    ViewData(const cv::cuda::GpuMat d_imColor, const cv::cuda::GpuMat d_imGray) {
        setView(d_imColor, d_imGray);
    }

    ViewData(const cv::Mat imColor, const cv::Mat imGray, const cv::cuda::GpuMat d_imColor, const cv::cuda::GpuMat d_imGray) {
        setView(imColor, imGray);
        setView(d_imColor, d_imGray);
    }
};

class View {
public:
    ViewData* viewPtr;

    void setView(ViewData* view) { this->viewPtr = view; }
};

class ViewDataContainer {
private:
    std::list<ViewData> m_dataContainer;
public:
    void addItem(ViewData viewData) { m_dataContainer.push_back(viewData); }

    ViewData* getLastOneItem() { return &*m_dataContainer.rbegin(); }

    ViewData* getLastButOneItem() { return &*++m_dataContainer.rbegin(); }
};

#endif //VIEW_H