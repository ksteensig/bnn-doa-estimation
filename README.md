# bnn-doa-estimation
Experimenting with Binarized Neural Networks for DoA estimation on 1-bit quantized signals. The benefit is that 1-bit ADCs are much simpler which is key in the case of massive MIMO systems. Enabling the use of thousands of receivers can give a much higher spatial resolution which is a corner stone in improving future wireless communication performance.

The purpose is to determine how 1-bit quantization of a received signal on a ULA performs compared to the unquantized during DoA estimation. The evidence shown by (One-bit MUSIC https://arxiv.org/pdf/1901.05109.pdf) suggests similar performance whether or not the signal has been quantized.

In case the BNN is not particularly worse than the DNN, it would be of interest to create a bitwise neural network using XNOR and popcount during inference. The only available FPGA I have is an iCE40 HX8k so a more powerful one is required if this is to be done.

# TODO
- [x] Implement data generation (room for improvement)
- [x] Define simple DNN and BNN
- [ ] Train DNN and BNN on the same data
- [ ] If possible, acquire FPGA powerful enough to run BNN
  - [ ] Implement BNN in a hardware description language and synthesize for aforementioned FPGA
- [ ] Clean up
- [ ] Experiment with existing DNN models instead of simple fully connected network

# Literature

* DoA parameter estimation with 1-bit quantization for MIMO  https://arxiv.org/pdf/1602.05462.pdf
* training binarized neural network: https://arxiv.org/pdf/1602.02830.pdf
* bitwise neural network model on FPGA http://arainhyy.github.io/proposal.html
* One-bit MUSIC https://arxiv.org/pdf/1901.05109.pdf
* Gaussian noise https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_AppendixA.pdf
