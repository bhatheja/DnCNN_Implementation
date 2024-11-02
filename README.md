## KLA Challenge (EE5179- Deep Learning for Imaging Course Project)

| Name | Roll No. |
|----------|----------|
| Rahul Bhatheja   | EE23M079   |
| Rohit S   | AM23S006   |
| Sourav Majhi | MA23M022 |
| Suman Das | MA23M023 |

### For this project, we implemented feed-forward denoising convolutional neural networks (DnCNN), which utilize residual learning and batch normalization to boost denoising performance. In addition to that, we used patching to augument dataset.

<a href="https://arxiv.org/abs/1608.03981" target="_blank">Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising</a>

## Model Architecture
![Architecture](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Model_architecure.png)

With This Model We could get resonably good psnr value with high noise even for less number of epochs
<h3>Results</h3>

| Image 1 | Image 2 |
|---------|---------|
| ![Image 1](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Psnr_vs_epoch.png) | ![Image 2](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR_Dist.png) |

![PSNR For Best Model](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/PSNR_for%20validation%20data.png)
#### Some Resultant denoised images 
![Result 1](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image1.png)
![Result 2](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image2.png)
![Result 3](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image_3.png)
![Result 4](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image_4.png)
![Result 5](https://github.com/bhatheja/DnCNN_Implementation/blob/main/images/Result_image_5.png)
