#ifndef __actionbuffer__
#define __actionbuffer__
#include "pybind11_common_header.hpp"
#include "typesdef.hpp"

class ActionBuffer
{
private:
    /* data */
    int mem_len, ptr;
    std::vector<std::vector<vector3d>> *buf; // action data
    std::vector<points_t> *out;

public:
    ActionBuffer(int mem_len);
    ~ActionBuffer();
    void setBuf(int Nrobot);
    void clear();
    void store(const posvels_t &posvels, const points_t &actions);
    void cal(const posvels_t &posvels);
    std::vector<points_t> get(const posvels_t &posvels);
};

ActionBuffer::ActionBuffer(int mem_len)
{
    this->mem_len = mem_len;
    this->ptr = 0;
    this->buf = nullptr;
    this->out = nullptr;
}

ActionBuffer::~ActionBuffer()
{
    delete this->buf;
    delete this->out;
}

void ActionBuffer::setBuf(int Nrobot)
{
    delete this->buf;
    delete this->out;
    this->buf = new std::vector<std::vector<vector3d>>(Nrobot, std::vector<vector3d>(this->mem_len));
    this->out = new std::vector<points_t>(Nrobot, points_t(this->mem_len));
    this->ptr = 0;
}

void ActionBuffer::clear()
{
}

void ActionBuffer::store(const posvels_t &posvels, const points_t &actions)
{
    for (size_t Nth = 0; Nth < posvels.size(); Nth++)
    {
        (*this->buf)[Nth][this->ptr][0] = actions[Nth][0];
        (*this->buf)[Nth][this->ptr][1] = actions[Nth][1];
        (*this->buf)[Nth][this->ptr][2] = posvels[Nth][2];
    }
    this->ptr = (this->ptr + 1) % this->mem_len;
}

void ActionBuffer::cal(const posvels_t &posvels)
{
    double rat_diff = 0;
    double cos_diff = 0, sin_diff = 0;
    for (size_t Nth = 0; Nth < posvels.size(); Nth++)
    {
        for (size_t mem_i = this->ptr; mem_i < this->ptr + this->mem_len; mem_i++)
        {

            rat_diff = posvels[Nth][2] - (*this->buf)[Nth][mem_i % this->mem_len][2];
            cos_diff = cos(rat_diff), sin_diff = sin(rat_diff);
            // TODO check the formula
            (*this->out)[Nth][mem_i - this->ptr][0] =
                cos_diff * (*this->buf)[Nth][mem_i % this->mem_len][0] + sin_diff * (*this->buf)[Nth][mem_i % this->mem_len][1];
            (*this->out)[Nth][mem_i - this->ptr][1] =
                -sin_diff * (*this->buf)[Nth][mem_i % this->mem_len][0] + cos_diff * (*this->buf)[Nth][mem_i % this->mem_len][1];
        }
    }
}

std::vector<points_t> ActionBuffer::get(const posvels_t &posvels)
{
    this->cal(posvels);
    return *this->out;
}

#endif //__actionbuffer__
