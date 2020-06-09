#ifndef VIEW_H
#define VIEW_H
#pragma once

#include "pch.h"

class ViewData {
public:
    cv::Mat imColor, imGray;

    ViewData() {}

    ViewData(cv::Mat imColor, cv::Mat imGray) {
        imColor.copyTo(this->imColor);
        imGray.copyTo(this->imGray);
    }
};

class View {
public:
    ViewData* viewPtr;

    void setView(ViewData* view) { this->viewPtr = view; }
};

/** 
 * ViewDataContainer to store views (images)
 * 
 * View components are referencing to ViewDataContainer
 */
class ViewDataContainer {
private:
    const uint m_containerBufferSize;

public:
    std::list<ViewData> m_dataContainer;
    /** ViewDataContainer constructor
     * 
     * @param containerBufferSize buffer size to clear memory
    */
    ViewDataContainer(const uint containerBufferSize = INT32_MAX)
        : m_containerBufferSize(containerBufferSize) {}

    void addItem(ViewData viewData) { 
        if (m_dataContainer.size() > m_containerBufferSize - 1) {
            std::list<ViewData> _dataContainer;

            _dataContainer.push_back(*++m_dataContainer.rbegin());
            _dataContainer.push_back(*m_dataContainer.rbegin());
            
            std::swap(m_dataContainer, _dataContainer);
        }  

        m_dataContainer.push_back(viewData); 
    }

    ViewData* getLastOneItem() { return &*m_dataContainer.rbegin(); }

    ViewData* getLastButOneItem() { return &*++m_dataContainer.rbegin(); }

    bool isEmpty() { return m_dataContainer.empty(); }
};

#endif //VIEW_H