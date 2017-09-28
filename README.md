# Carvana-Image-Masking-Challenge
CNN Architecture which achieved 99.55% accuracy in the Carvana Image Masking Challenge.
Link for challenge: https://www.kaggle.com/c/carvana-image-masking-challenge
My contestant name: AntiLippasaar

# Content

model_03.ipynb - contains the logic and model architecture, runnings this file will give you the models needed to achieve accuracy of 99.55%.

save_submission.ipynb - this saves the predictions so you can make multiple predictions from different models and then combine them as one.

Combine.ipynb - this is used to combine predictions of multiple models into one.

to_csv.ipynb - saves the results in the expected form and shape, so you can submit them to the competition

anti - This folder contains some functions for getting the data, data augmentation, threading etc. Meant to use for this competition.

# Achieve 99.99.55%

For that I ran the model_03.ipynb on 1080ti for about 2 days. To get 99.55% I made predictions for the 5 best models and then combined them into 1 predicition using Combine.ipynb
