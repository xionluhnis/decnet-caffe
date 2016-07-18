#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {


template <typename Dtype>
void SelectSegBinaryTwoFramesLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_gpu_data());
  caffe_copy(this->prefetch_data2_.count(), this->prefetch_data2_.cpu_data(),
             top[1]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[2]->mutable_gpu_data());
  }
  if (this->output_data_dim_) {
    caffe_copy(this->prefetch_data_dim_.count(), this->prefetch_data_dim_.cpu_data(),
               top[3]->mutable_gpu_data());
  }

  // Start a new prefetch thread
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}


INSTANTIATE_LAYER_GPU_FORWARD(SelectSegBinaryTwoFramesLayer);

}  // namespace caffe
