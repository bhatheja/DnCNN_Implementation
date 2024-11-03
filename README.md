## KLA Challenge (EE5179- Deep Learning for Imaging Course Project)

| Name | Roll No. |
|----------|----------|
| Rahul Bhatheja   | EE23M079   |
| Rohit S   | AM23S006   |
| Sourav Majhi | MA23M022 |
| Suman Das | MA23M023 |

### For this project, we implemented feed-forward denoising convolutional neural networks (DnCNN), which utilize residual learning and batch normalization to boost denoising performance. In addition to that, we used patching of 128x128 to augument dataset.

<a href="https://arxiv.org/abs/1608.03981" target="_blank">Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising</a>

## Model Architecture
![Architecture](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Model_architecure.png)

This Model provides resonably good psnr value with high noise even when trained for less number of epcohs and large enough patching.
<h3>Results. We had choosen the depth of the network to be 17 which is large enough for capturing the contextual information from the surrounding of the pixel. The receptive field for the implemented netword is 35 for out case.</h3>




### Few Resultant denoised images for demonstration
![Result 1](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image1.png)
![Result 2](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image2.png)
![Result 3](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image_3.png)
![Result 4](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image_4.png)
![Result 5](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image_5.png)

### Quantitative Analysis for the validation data
#### Convergence of PSNR and SSIM:
| PSNR VS EPOCH | SSIM VS EPOCH |
|---------|---------|
| ![PSNR VS EPOCH](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR_vs_Epoch.png) | ![SSIM VS EPOCH](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/SSIM_vs_Epoch.png) |


#### Parameters Distribution Across Epochs:
| PSNR Distribution | SSIM Distribution |
|---------|---------|
| ![PSNR Distribution](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR_Dist_Across_Epoch.png) | ![SSIM Distribution](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/SSIM_Dist_Across_Epoch.png) |


#### Distribution of PSNR and SSIM for the best model:
| PSNR Frequency Hist | SSIM Frequency Hist |
|---------|---------|
| ![PSNR Frequency Hist](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR_Hist.png) | ![SSIM Frequency Hist](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/SSIM_Hist.png) |

### Masked defect
![masked_1](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/masked_1.png)
![masked_2](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/masked_2.png)
![masked_3](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/masked_3.png)
### Evaluation after masking defect in clean image and denoised image
#### SSIM for the complete image
![SSIM](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/SSIM.png)

#### PSNR for the complete image which will generally be very high becuase size of the mask is very small
![PSNR](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR.png)

#### PSNR value for only active pixels
![PSNR Active](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR_ACTIVE.png)

| Acknowledgements |
| ---------------- |
| <a href="https://arxiv.org/abs/1608.03981" target="_blank">Paper on DnCNN by  Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang</a>|
| Kaggle for their free gpu access |

