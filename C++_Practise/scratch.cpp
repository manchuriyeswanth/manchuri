#include<iostream>
#include<vector>
#include<memory>
#include<bits/stdc++.h> 
using namespace std;

class Tensor
{
    public:
    Tensor(size_t size):size_(size)
    {
        std::cout<<"Allocating Tensor of size "<<size<<endl;
        data_ = std::make_unique<float[]>(size);
    }
    ~Tensor(){
        cout<<"Destroying Size"<<endl;
    }

    float* data() const
    {
        return data_.get();
    }
    size_t size() const
    {
        return size_;
    }
    private:
    size_t size_;
    std::unique_ptr<float[]> data_;
};

int main()
{
    Tensor t(2);
    return 0;
}