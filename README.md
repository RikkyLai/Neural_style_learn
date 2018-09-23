# Neural style learn
I just happened to learn Neural network recently and was interested in this project, which can convert a noisy image into a fusion image inputting a contest image and a stylized image. I have searched several blogs and finally got one which help me easily realize this effect without complex codes. So I want to write it down in case I forget some details in the future.



## Network
Here we are using vgg19 as the network, you can download it through [imagenet-vgg-verydeep-19](http://www.vlfeat.org/matconvnet/models/beta16/). You need to know every layer's name so you can get all the results after the input image get through the layers. It is important because these results would form the loss function you need later. 


```
layers = 
(
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
```
of course you need to know the structure of the network you have download. Weights and bias could be got through this code `weights, bias = data['layers'][0][i][0][0][0][0]`. I strongly recommand you to test the structure.

We don't train any weights and bias. The only thing we train is the noisy image.

The inputs of Network have to be normalized.


## Stylize
We get a content image and a style image, both of which should be imported into the network and we have to record all hidden layers' results. Also, we create a target noisy image which is going to mix the content and the style, being imported to the network and get the results. 

The most important thing or the fanciest thing in my view is these results would form the loss. 

details in [paper](https://arxiv.org/abs/1508.06576)


squared-error loss both in content loss and style loss

the difference is gram matrix


