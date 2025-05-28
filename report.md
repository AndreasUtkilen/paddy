# Assigment 3

## Task 1 - Paddy Disease Detection

As stated in the assignment, we will create a model to detect paddy diseases using the dataset provided. I started by looking and implementing the code that Jeremy Howard provided, and managed to get similar results to the ones he got, although the provided compute is not suitable for his models and resulted in a very long training time. I then started to play with the model and tried to improve the results.

The first thing I did was to try different architectures. I started with the Resnet50 architecture, which is a very popular architecture for image classification tasks, and to reduce the training time, I used a smaller input size of 224x224. The results were not very good, with a validation accuracy of around 0.8. I then tried the Resnet101 architecture, which is a deeper version of Resnet50, and the results improved slightly, with a validation accuracy of around 0.86.

I then tried the ConvNext Small 22k and found that it performed much better than a Resnet101 or similar architectures. This is likely due to the fact that ConvNext is a modernized convolutional neural network inspired by the transformer era (like ViT), while it keeps the efficiency of CNNs and that the Convnext model is pretrained on ImageNet-22K (22,000 classes), and Resnet is usually pretrained on ImageNet-1K (1,000 classes). ConvNext builds stronger hierarchical features due to its modern block structure and larger receptive fields, and since the disease is not very visible, the model needs to learn more complex features. 

To make the model even better I created an ensemble of Resnet models and ConvNext models with different hyperparameters and augmentations. The results were very good, with a score of around 0.97846 on Kaggle, while still keeping the training time to a reasonable level (30 min, 8 GB GPU memory). The final ensemble looked like this:

| Arch                  | Image size | Epochs | Augmentations   | Error rate |
| --------------------- | ---------- | ------ | --------------- | ---------- |
| resnet50              | 224        | 10     | scale (min 75%) | 0.144      |
| resnet101             | 224        | 10     | scale (min 75%) | 0.142      |
| convnext_small_in22k  | 224        | 10     | scale (min 75%) | 0.028      |
| vit_small_patch16_224 | 224        | 10     | scale (min 75%) | 0.026      |

The ensemble works similarly to Jeremy's model, where the predictions of each model are averaged to get the final prediction.

After realizing that the resnets were not performing as well as the ConvNext and ViT models, I decided to remove them from the ensemble and only keep the ConvNext and ViT models. The final ensemble was then:

| Arch                  | Image size | Epochs | Augmentations   | Error rate |
| --------------------- | ---------- | ------ | --------------- | ---------- |
| convnext_small_in22k  | 224        | 10     | scale (min 75%) | 0.024      |
| vit_small_patch16_224 | 224        | 10     | scale (min 75%) | 0.029      |

Which resulted in a score of 0.98306 on Kaggle, which is slightly better. An ensemble with only two models is not really enough when the descision is based on the average of the predictions, so I added two more variants of the ConvNext model and the ViT model with different augmentations and hyperparameters. The ensemble looked like this:

| Arch                  | Image size | Epochs | Augmentations   | Error rate |
| --------------------- | ---------- | ------ | --------------- | ---------- |
| convnext_small_in22k  | 299        | 10     | scale (min 75%) | 0.024      |
| convnext_small_in22k  | 224        | 10     | scale (min 75%) | 0.024      |
| vit_small_patch16_224 | 299        | 10     | scale (min 75%) | 0.029      |
| vit_small_patch16_224 | 224        | 10     | scale (min 75%) | 0.029      |

This resulted in a score of 0.98385 on Kaggle. The training time was around 30 minutes, and the GPU memory usage was around 8 GB.

As a final test, I added a nother ConvNext model with a different input size of 320. Which resulted in a slightly higher score (0.98539) on Kaggle. The final ensemble looked like this:

| Arch                  | Image size | Epochs | Augmentations   | Error rate |
| --------------------- | ---------- | ------ | --------------- | ---------- |
| convnext_small_in22k  | 320        | 10     | scale (min 75%) | 0.021      |
| convnext_small_in22k  | 299        | 10     | scale (min 75%) | 0.024      |
| convnext_small_in22k  | 224        | 10     | scale (min 75%) | 0.025      |
| vit_small_patch16_224 | 244        | 10     | scale (min 75%) | 0.032      |

The final ensemble increased the training time to around 60 minutes.


## Task 2 - Deepfake Detection

In this task, we will create a model to detect deepfake videos using a dataset of real and fake images. Similar to the paddy disease detection task, we will leverage transfer learning and fine-tune a pre-trained model for this purpose.

I started by looking at the dataset and applied the same preprocessing steps as in the paddy disease detection task. I used the resnet architecture with few epochs, which makes training very fast. It was then possible to quickly iterate on the process in terms of augmentations and hyperparameter tuning. 

I ran a lot of experiments with different image input sizes and epochs to find the best resnet model. I started with 512x512 before testing 256x256 and finally 128x128. I found that the smaller the image size, the better the model performed, with limited training time.

The resnets performeance was not very good, and resulted in a Kaggle score of around 0.86. I then tried the ConvNext Small 22k and found that it performed much better than the resnet models, with a score of around 0.92.

After researching the topic, I found a paper called "Fighting deepfake by exposing the convolutional traces on images" (Guarnera et al., 2020) that suggested using a technique were you calculate convolution traces of the images, by using a algorithm called Expectation Maximization (EM). A simple implementation of this algorithm was tested, and by using the Random Forest classifier, as recommended in the paper, I managed to get a score of around 0.95 on Kaggle.


Guarnera, L., Giudice, O., & Battiato, S. (2020). Fighting deepfake by exposing the convolutional traces on images. IEEE access, 8, 165085-165098.