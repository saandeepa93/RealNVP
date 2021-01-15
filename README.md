# **RealNVP based training for MNIST and celeba dataset**

## **Setup**
```
pip install -r requirements.txt
```

## **Training**
```
python train.py --no_cuda False --root root_path
```

The architecture has been taken from the official implementation from [tensorflow](https://github.com/tensorflow/models/tree/archive/research/real_nvp)
The inference after each epoch is generated under the *sample* directory

## **Samples**

### *MNIST*
<img src="./samples/9_234_3488.948.png">
<img src="./samples/97_234_864.647.png">
<img src="./samples/199_234_860.3770000000001.png">

### *Celeba*
<img src="./samples/celeba.png">
<img src="./samples/33_256_7622.09.png">
