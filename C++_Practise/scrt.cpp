#include <iostream>
#include <memory>

using namespace std;

class Tensor
{
public:
    Tensor(size_t size) : size_(size)
    {
        cout << "Allocating Size " << size << endl;
        data_ = make_unique<float[]>(size);
    }

    ~Tensor()
    {
        cout << "Destroying Size" << endl;
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
    unique_ptr<float[]> data_;
};

class Model
{
public:
    Model()
    {
        cout << "Model Initialized" << endl;
    }

    float forward(const Tensor& input)
    {
        float sum = 0;
        for (size_t i = 0; i < input.size(); i++)
        {
            sum += input.data()[i];
        }
        return sum;
    }
};

class InferenceEngine
{
public:
    InferenceEngine(size_t buffersize)
    {
        buffer_ = make_unique<Tensor>(buffersize);
        model_ = make_unique<Model>();
        cout << "Inference Model Initialized" << endl;
    }

    float run()
    {
        for (size_t i = 0; i < buffer_->size(); i++)
        {
            buffer_->data()[i] = static_cast<float>(i);
        }
        return model_->forward(*buffer_);
    }

private:
    unique_ptr<Tensor> buffer_;
    unique_ptr<Model> model_;
};

int main()
{
    InferenceEngine I(10);
    float out = I.run();
    cout << "Output: " << out << endl;
    return 0;
}
