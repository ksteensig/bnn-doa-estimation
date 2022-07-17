# bnn-doa-estimation
Experimenting with Binarized Neural Networks for DoA estimation on 1-bit quantized signals

* DoA parameter estimation with 1-bit quantization for MIMO  https://arxiv.org/pdf/1602.05462.pdf
* training binarized neural network: https://arxiv.org/pdf/1602.02830.pdf
* bitwise neural network model on FPGA http://arainhyy.github.io/proposal.html
* Gaussian noise https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_AppendixA.pdf

# TODO
- [x] Implement data generation (room for improvement)
- [x] Define DNN and BNN
- [ ] Train DNN and BNN on the same data
- [ ] If possible, acquire FPGA powerful enough to run BNN
  - [ ] Implement BNN in a hardware description language and synthesize for aforementioned FPGA
