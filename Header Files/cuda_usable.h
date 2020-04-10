#ifndef CUDA_USABLE_H
#define CUDA_USABLE_H
#pragma once

class CUDAUsable {
protected:
    bool m_isUsingCUDA;
public:
    bool getIsUsingCUDA() const { return m_isUsingCUDA; }
};

#endif //CUDA_USABLE_H