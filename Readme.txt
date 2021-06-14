	Deep Convolutional Neural Network Based on Attention Mechanism of Inter-layer Relationship
	(Copyright 2021 by Xiaofei-Wang, Zhejiang University of Technology, All rights reserved)

1.What is Attention Mechanism of Inter-layer Relationship?
	Attention Mechanism of Inter-layer Relationship is a channel attention mechanism based on the relationship between convolutional layers.
	
2.How to configure the environment?

	Anaconda is recommended to set up Pytorch environments
	Python3.7 with numpy and numba should be installed
	Pytorch-1.5.1 (https://pytorch.org/get-started/previous-versions/) 
	cuda = 10.2
	
3.Datasets.
	Classification dataset: 
		Cifar10 (http://www.cs.toronto.edu/~kriz/cifar.html    Version : CIFAR-10 python version)
		Cifar100 (http://www.cs.toronto.edu/~kriz/cifar.html    Version : CIFAR-100 python version)
	
	Semantic segmentation datasets: 
		VOC2007 (https://www.kaggle.com/zaraks/pascal-voc-2007)
		VOC2012 (https://www.kaggle.com/huanghanchina/pascal-voc-2012)
		COCO2017 (https://cocodataset.org/#download)
		The image name (* txt file) of the used dataset and the processing method 
for converting the COCO dataset into a semantic partitioned dataset are placed in the data_deal folder.
	
	Medical image segmentation datasets:
		ISIC2018 (It is available in the ISIC2018_datasets folder)
		Liver （It is available in the Liver_datasets folder）

4. Experimental code
	The codes of the three tasks are uploaded separately for reference.
	
