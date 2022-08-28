# Mnist-in-TensorFlow
In tensorflow 2.0, it seems that there is no mnist anymore.

Here is the solutions for the beginners who wants to study Tensorflow with books using tutorials(Mnist) and tensorflow 1.x

The .zip file is the original tutorials dictionary. Download it and unzip it

Create a new dictionary called examples in the tensorflow dictionary, and copy the tutorials file into the examples dictionary

Then 'from tensorflow.examples.tutorials.mnist import input_data' will work

If you don't know your tensorflow path. Open your cmd and type, pip show tensorflow, the  you can see your path

![image](https://user-images.githubusercontent.com/84928349/187093835-721ceed8-5caf-4799-811c-790ab144c2cd.png)

I also provide two example codes for Mnist from my book to help testing your environment

IMPORTENT THINGS ABOUT THE EXAMPLE CODE:

The whole environment is under tensorflow 1.0. I installed tensorflow 2.0, and there is a way to make it become 1.0

import tensorflow._api.v2.compat.v1 as tf

Aboving code transfer 2.0 to 1.0

tf.disable_v2_behavior()

Aboving code disable v2's environment
