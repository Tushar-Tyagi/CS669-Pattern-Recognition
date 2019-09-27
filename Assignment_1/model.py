'''
	To make a model.

'''

import numpy as np

def compile(train_data1, train_data2, train_data3, case):

	mean1 = np.mean(train_data1, axis = 0)
	mean2 = np.mean(train_data2, axis = 0)
	mean3 = np.mean(train_data3, axis = 0)

	cov1 = np.cov(train_data1, rowvar=False)
	cov2 = np.cov(train_data2, rowvar=False)
	cov3 = np.cov(train_data3, rowvar=False)

	meanVar1 = np.mean(cov1)
	meanVar2 = np.mean(cov2)
	meanVar3 = np.mean(cov3)
	meanVar = (meanVar1 + meanVar2 + meanVar3)/3

	if case == 1:
		#Covariance matrix for all the classes is the same and is σ2I
		cov1 = np.diag(np.diag(cov1))
		cov2 = np.diag(np.diag(cov2))
		cov3 = np.diag(np.diag(cov3))

		return [[mean1,cov1],[mean2,cov2],[mean3,cov3]]

	elif case == 2:
		#Full Covariance matrix for all the classes is the same and is Σ.

		cov = (cov1+cov2+cov3)/3
		cov1=cov
		cov2=cov
		cov3=cov
		
		return [[mean1,cov1],[mean2,cov2],[mean3,cov3]]

	elif case == 3:
		#Covariance matric is diagonal and is different for each class

		for i in range(cov1.shape[0]): #rows
			for j in range(cov1.shape[1]): #columns
				if i!=j:
					cov1[i][j]=0
		for i in range(cov2.shape[0]):
			for j in range(cov2.shape[1]):
				if i!=j:
					cov2[i][j]=0
		for i in range(cov2.shape[0]):
			for j in range(cov2.shape[1]):
				if i!=j:
					cov2[i][j]=0

		return [[mean1,cov1],[mean2,cov2],[mean3,cov3]]

	else:
		#Full Covariance matrix for each class is different
		return [[mean1,cov1],[mean2,cov2],[mean3,cov3]]

def valCalc(data, classPara, classPriori):
	#classPara is a list: 1st-mean of class, 2nd-cov matrix of class
	#In this assignment, number of dimensions = 2, so:
	d=2

	posterior = 1/((np.power(2*np.pi,d/2))*(np.sqrt((np.linalg.det(classPara[1])))))
	posterior = posterior*np.exp(-0.5*(np.matmul(np.matmul(data-classPara[0],np.linalg.inv(classPara[1])),(data-classPara[0]))))
	return posterior*classPriori

def test(testData1, testData2, testData3, parameters, classPriories):
	#parameters: list of lists.
	#classPriories: list of Priories of classes

	class1TP=0
	class2TP=0
	class3TP=0
	class1TN=0
	class2TN=0
	class3TN=0
	class1FP=0
	class2FP=0
	class3FP=0
	class1FN=0
	class2FN=0
	class3FN=0
	#Accuracy = total true positives/ total samples
	#Precesion of a class = True Positives/TPs+FPs

	for i in testData1:
		p1 = valCalc(i, parameters[0], classPriories[0])
		p2 = valCalc(i, parameters[1], classPriories[1])
		p3 = valCalc(i, parameters[2], classPriories[2])

		if p1>p2 and p1>p3:
			class1TP+=1
			class2TN+=1
			class3TN+=1
		elif p2>p3 and p2>p1:
			class1FN+=1
			class2FP+=1
			class3TN+=1
		else:
			class1FN+=1
			class2TN+=1
			class3FP+=1

	for i in testData2:
		p1 = valCalc(i, parameters[0], classPriories[0])
		p2 = valCalc(i, parameters[1], classPriories[1])
		p3 = valCalc(i, parameters[2], classPriories[2])

		if p2>p1 and p2>p3:
			class2TP+=1
			class1TN+=1
			class3TN+=1
		elif p1>p2 and p1>p3:
			class2FN+=1
			class1FP+=1
			class3TN+=1
		else:
			class2FN+=1
			class1TN+=1
			class3FP+=1

	for i in testData3:
		p1 = valCalc(i, parameters[0], classPriories[0])
		p2 = valCalc(i, parameters[1], classPriories[1])
		p3 = valCalc(i, parameters[2], classPriories[2])

		if p3>p1 and p3>p2:
			class3TP+=1
			class1TN+=1
			class2TN+=1
		elif p2>p3 and p2>p1:
			class3FN+=1
			class2FP+=1
			class1TN+=1
		else:
			class3FN+=1
			class1FP+=1
			class2TN+=1

	totalData = testData1.shape[0]+testData2.shape[0]+testData3.shape[0]

	return [[class1TP,class1TN,class1FN,class1FP],[class2TP,class2TN,class2FN,class2FP],[class3TP,class3TN,class3FN,class3FP],totalData]



