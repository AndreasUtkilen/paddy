# Assigment 3

## Task 1 - Paddy Disease Detection

As stated in the assignment, we will create a model to detect paddy diseases using the dataset provided. I started by looking and implementing the code that Jeremy Howard provided, and managed to get similar results to the ones he got, although the provided compute is not suitable for his models and resulted in a very long training time. I therefore started to play with the model and tried to improve the results while keeping the training time at tolerable level.

I initially did a test run of what Jeremy Howard had done, just to see if I could replicate his results. After several hours of training on a Google Colab notebook, which offers 16 GB of VRAM compared to our 8 GB at the computer lab, I got a similar result, but was left with no more compute tokens. I therefore realised that in order to continue working on this task with limited compute I had to make some changes.

The first thing I did was to try different architectures. I started with the Resnet50 architecture, which is a very popular, but now an old architecture for image classification tasks, and to reduce the training time, I used a smaller input size of 224x224. The results were not very good, with a validation accuracy of around 0.8. I then tried the Resnet101 architecture, which is a deeper version of Resnet50, and the results improved slightly, with a validation accuracy of around 0.86. I could have tried to improve these resnet models more, with more augmentations and fine-tuning the hyperparameters, but I knew that Mr. Howard has used different architectures, such as ConvNext and ViT.

I therefore tried the ConvNext Small 22k and found that it performed much better than a Resnet101 and similar architectures. This is likely due to the fact that ConvNext is a modernized convolutional neural network inspired by the transformer era (like ViT), while it keeps the efficiency of CNNs and that the Convnext model is pretrained on ImageNet-22K (22,000 classes), and Resnet is usually pretrained on ImageNet-1K (1,000 classes). The ConvNext has a tendency to build stronger hierarchical features due to its modern block structure and larger receptive fields, and since the disease is not very visible, the model needs to learn more complex features.

To try to make the model even better I created an ensemble of Resnet models and ConvNext models with different hyperparameters and augmentations. The results were very good, with a score of around 0.97846 on Kaggle, while still keeping the training time to a reasonable level (34 min, 8 GB GPU memory). The first ensemble looked like this:

| Arch                  | Image size | Epochs | Error rate | Training time (approx) |
| --------------------- | ---------- | ------ | ---------- | ---------------------- |
| resnet50              | 224        | 10     | 0.144      | 7 min                  |
| resnet101             | 224        | 10     | 0.142      | 10 min                 |
| convnext_small_in22k  | 224        | 10     | 0.028      | 11 min                 |
| vit_small_patch16_224 | 224        | 10     | 0.026      | 6 min                  |

The ensemble works similarly to Jeremy's model, where the predictions of each model are averaged to get the final prediction.

After realizing that the resnets were not performing as well as the ConvNext and ViT models, I decided to remove them from the ensemble and only keep the ConvNext and ViT models. The next ensemble turned out like this:

| Arch                  | Image size | Epochs | Error rate | Training time (approx) |
| --------------------- | ---------- | ------ | ---------- | ---------------------- |
| convnext_small_in22k  | 224        | 10     | 0.024      | 11 min                 |
| vit_small_patch16_224 | 224        | 10     | 0.029      | 6 min                  |

Which resulted in a score of 0.98306 on Kaggle, which is slightly better. An ensemble with only two models is not really enough when the descision is based on the average of the predictions though, so I added two more variants of the ConvNext model and the ViT model with different augmentations and hyperparameters. The improved ensemble looked like this:

| Arch                  | Image size | Epochs | Error rate | Training time (approx) |
| --------------------- | ---------- | ------ | ---------- | ---------------------- |
| convnext_small_in22k  | 299        | 10     | 0.024      | 16 min                 |
| convnext_small_in22k  | 224        | 10     | 0.024      | 11 min                 |
| vit_small_patch16_224 | 224        | 10     | 0.029      | 6 min                  |

This resulted in a score of 0.98385 on Kaggle. The training time was around 30 minutes, and the GPU memory usage was around 8 GB.

As a final test, I added a nother ConvNext model with a different input size of 320. Which resulted in a slightly higher score (0.98539) on Kaggle, but with the downside of a substantially longer training time of around 1 hour and 34 minutes. The final ensemble looked like this:

| Arch                  | Image size | Epochs | Error rate | Training time (approx) |
| --------------------- | ---------- | ------ | ---------- | ---------------------- |
| convnext_small_in22k  | 320        | 10     | 0.021      | 60 min                 |
| convnext_small_in22k  | 299        | 10     | 0.024      | 16 min                 |
| convnext_small_in22k  | 224        | 10     | 0.025      | 11 min                 |
| vit_small_patch16_224 | 244        | 10     | 0.032      | 6 min                  |

I spent a lot of time trying out if I could improve the GPU/CPU usage by rescaling the whole dataset to the desired sizes before starting any training, without much success. The resize function and scale augmentations in Fast AI are not the bottle neck of the training. 

In terms of other augmentations, I used Fast AI's standard augmentations which is defined as follows:

| Augmentation                           | Value/Setting         |
|----------------------------------------|-----------------------|
| Random horizontal flips                | Yes                   |
| Maximum degree of rotation             | 10°                   |
| Minimum zoom                           | 1.0                   |
| Maximum zoom                           | 1.1                   |
| Maximum scale for changing brightness  | 0.2                   |
| Maximum value for changing warp        | 0.2                   |
| Probability of affine transformation   | 0.2                   |
| Probability of brightness/contrast     | 0.75                  |
| Padding mode                           | Reflection            |


## Task 2 - Deepfake Detection

In this task, we will create a model to detect deepfake images using a dataset of real and fake images. Similar to the paddy disease detection task, we will leverage transfer learning and fine-tune a pre-trained model for this purpose.

I started by looking at the dataset and applied the same preprocessing steps as in the paddy disease detection task. I used the resnet50 architecture with few epochs, which makes training very fast. It was then possible to quickly iterate on the process in terms of augmentations and hyperparameter tuning. I ran a lot of experiments with different image input sizes and epochs to find the best resnet model. I started with 512x512 before testing 256x256 and finally 128x128. I found that the smaller the image size, the better the model performed, but when the images becomes too small they loose the subtle errors and noise that distinguish the real from the fake iamges. The resnet50s performance was satisfactory, but not very good, and resulted in a Kaggle score of around 0.895.

I then explored other architectures such as ConvNext and ViT to see if those worked better on this challenge. 

Similarly to the paddy disease task, creating an ensemble of different architectures should improve the final model as it hedges on the models good qualities.

After researching the topic, I found a paper called "Fighting deepfake by exposing the convolutional traces on images" (Guarnera et al., 2020) that suggested using a technique were you calculate convolution traces of the images, by using a algorithm called Expectation Maximization (EM). The method works by calculating the convolution traces of the images, which are the tiny patterns, imperfections and artifacts left behind by the GANs (Generative Adversarial Networks) that create deepfakes. These traces are often too subtle for the human eye to detect, but they can be captured by the EM algorithm. The EM algorithm iteratively refines the estimates of the convolution traces until they converge to a stable solution. Once the traces are calculated, they can be used as features for a classifier, such as Random Forest, to distinguish between real and fake images. A simple implementation of this algorithm was tested, and by using the Random Forest classifier, as recommended in the paper, I managed to get a accuracy of around 0.6, which is not particularly good. I therefore tried to use Support Vector Classification (SVC) as an alternative to Random Forest classifier. This increased the accuracy to 0.7, and shows that the method works, as the computed convolution traces have some signal, but they are noisy and hard to classify. In the paper, they get an even higher accuracy at around 0.9 and above, which tells me that my implementation of the paper might not be as good as theirs, and without GPU acceleration of the convolution trace computation it takes many hours to finish the dataset. 

Looking at this solution it is quite nice to see that the task is solvable without needing a black box, which these CNNs and RNNs are, and it is possible to explain how the model made its prediction on a understandable level. Making an understandable model will be important to be able to gain trust in the model and will supersede any deep learning model with equal accuracy.

Guarnera, L., Giudice, O., & Battiato, S. (2020). Fighting deepfake by exposing the convolutional traces on images. IEEE access, 8, 165085-165098.