# Real-World Masked Face Dataset（RMFD）

Because of the recent epidemic of COVID-19 virus around the world, people across the country wear masks and there appear a large number of masked face samples. We thus created the world's largest masked face dataset to accumulate data resources for possible intelligent management and control of similar public safety events in the future. Based on masked face dataset, corresponding masked face detection and recognition algorithms are designed to help people in and out of the community when the community is closed. In addition, the upgrade of face recognition gates, facial attendance machines, and facial security checks at train stations is adapted to the application environment of pedestrian wearing masks.

Sponsor: National Engineering Research Center for Multimedia Software (NERCMS), School of Computer Science, Wuhan University

Contact: Zhangyang Xiong, Email: x_zhangyang@whu.edu.cn

In order to further expand this dataset, everyone is welcome to send personally collected pictures of masks to x_zhangyang@whu.edu.cn by email, and we will process the received pictures uniformly.

## Download Datasets

Part of the original samples has been uploaded to this github website. RFMD_part_1 can be downloaded directly. RFMD_part_2 (4 compressed files) and RFMD_part_3 (3 compressed files) need to download all compressed files before decompressing them.You can also download these datasets from the link below.

Download link: https://pan.baidu.com/s/1Vly3K-0qjlB6M2lenTZ8PA Password: xhze 

or https://drive.google.com/open?id=1kZAIiv34Iav9Vt8BB101FXo4KoEClpx9

More labeled face samples are illustrated as follows: (different from raw samples in github) Different from the facial mask recognition (or detection) dataset, the masked face recognition dataset must include multiple masked and unmasked face images of the same subject. To this end, we have established two kinds of masked face recognition datasets. 

(1)	Real-world masked face recognition dataset: We crawled the samples from the website. After cleaning and labeling, it contains 5,000 masked faces of 525 people and 90,000 normal faces. 

Download link: https://pan.baidu.com/s/1XvGepj84SCA9rlVb9rGhEQ  Password: j3aq

or https://drive.google.com/open?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp

(2)	Simulated masked face recognition datasets: We put on the masks on the faces in the public face datasets, and obtained the simulated masked face dataset of 500,000 faces of 10,000 subjects.

WebFace simulated masked face dataset:
https://pan.baidu.com/s/1Qi_8D_kH2QCm761elZs5YA Password: 77m8

or https://drive.google.com/open?id=1q0ibaoFVEmXrjlk3-Oyx2oYR8HpVy6jc

LFW simulated masked face dataset:
https://pan.baidu.com/s/1Ge0KcYgu6oVAbLlDHCKwRg Password: o126

or https://drive.google.com/open?id=1soLIUkGruSKMzg5z5_OYYqUVoca4E_lI

## Masked Face Recognition

Based on the constructed datasets, we designed and trained a face-eye-based multi-granularity masked face recognition model. The face identification accuracy on the dataset is over 95%, and some real-time video demos are as follows:

Download link: https://pan.baidu.com/s/1P0PiWFNT1z_TcCj8vo43ow Password: acwe 

![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/demo/wnx.gif)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/demo/wuhao.gif)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/demo/hzb.gif)


## Related Work

https://arxiv.org/abs/2003.09093
 
## Examples of Raw Samples

![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/RWMFD_part_1/0000/0003.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/RWMFD_part_1/0000/0001.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/RWMFD_part_1/0000/0002.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/example/0.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/example/1.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/example/2.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/example/3.jpg)
![image](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset/blob/master/example/4.jpg)
