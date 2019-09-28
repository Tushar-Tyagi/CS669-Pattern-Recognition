import numpy as np
import os
import urllib.request
import model
import plot

def main():
	data1_1 = np.loadtxt(fname = './Data/LS_Group04/Class1.txt')
	data1_2 = np.loadtxt(fname = './Data/LS_Group04/Class2.txt')
	data1_3 = np.loadtxt(fname = './Data/LS_Group04/Class3.txt')

	data2 = np.loadtxt(fname = './Data/NLS/data3.txt')#Class 1: 2446 , Class 2: 2447
	data2_1 = data2[0:300]
	data2_2 = data2[300:800]
	data2_3 = data2[800:1800]

	data3_1 = np.loadtxt(fname = './Data/rd_group2/class1.txt')
	data3_2 = np.loadtxt(fname = './Data/rd_group2/class2.txt')
	data3_3 = np.loadtxt(fname = './Data/rd_group2/class3.txt')

	data1_1x, data1_1y = data1_1[:(int)((data1_1.shape[0])*0.75)], data1_1[int((data1_1.shape[0])*0.75):]
	data1_2x, data1_2y = data1_2[:(int)((data1_2.shape[0])*0.75)], data1_2[int((data1_2.shape[0])*0.75):]
	data1_3x, data1_3y = data1_3[:int((data1_3.shape[0])*0.75)], data1_3[int((data1_3.shape[0])*0.75):]

	data2_1x, data2_1y = data2_1[:int((data2_1.shape[0])*0.75)], data2_1[int((data2_1.shape[0])*0.75):]
	data2_2x, data2_2y = data2_2[:int((data2_2.shape[0])*0.75)], data2_2[int((data2_2.shape[0])*0.75):]
	data2_3x, data2_3y = data2_3[:int((data2_3.shape[0])*0.75)], data2_3[int((data2_3.shape[0])*0.75):]

	data3_1x, data3_1y = data3_1[:int((data3_1.shape[0])*0.75)], data3_1[int((data3_1.shape[0])*0.75):]
	data3_2x, data3_2y = data3_2[:int((data3_2.shape[0])*0.75)], data3_2[int((data3_2.shape[0])*0.75):]
	data3_3x, data3_3y = data3_3[:int((data3_3.shape[0])*0.75)], data3_3[int((data3_3.shape[0])*0.75):]

	promptData = "Enter data choice:\n[L]inearly Saperable Data\n[N]on-Linearly Saperable Data\n[R]eal Data\n"
	promptCase = "Enter case choice:\n[1]Covariance matrix for all the classes is the same and is σ2I\n[2]Full Covariance matrix for all the classes is the same and is Σ.\n[3]Covariance matric is diagonal and is different for each class\n[4]Full Covariance matrix for each class is different\n"

	data_choice = input(promptData)
	case_choice = int(input(promptCase))

	if case_choice!=1 and case_choice!=2 and case_choice!=3 and case_choice!=4:
		print('Wrong case choice.\n Exiting')
		exit(0)
	
	#Accuracy = total true positives/ total samples
	#Precision of a class = True Positive/TPs+FPs
	#Recall of a class = true Positives/ TPs + FNs
	#F-Score of a class = harmonic mean of recall and precision = 2*(recall*precision)/(recall+precision)

	priori1_1 =	data1_1x.shape[0]/(data1_1x.shape[0]+data1_2x.shape[0]+data1_3x.shape[0])
	priori1_2 = data1_2x.shape[0]/(data1_1x.shape[0]+data1_2x.shape[0]+data1_3x.shape[0])
	priori1_3 = data1_3x.shape[0]/(data1_1x.shape[0]+data1_2x.shape[0]+data1_3x.shape[0])
	priori2_1 =	data2_1x.shape[0]/(data2_1x.shape[0]+data2_2x.shape[0]+data2_3x.shape[0])
	priori2_2 =	data2_2x.shape[0]/(data2_1x.shape[0]+data2_2x.shape[0]+data2_3x.shape[0])
	priori2_3 =	data2_3x.shape[0]/(data2_1x.shape[0]+data2_2x.shape[0]+data2_3x.shape[0])
	priori3_1 = data3_1x.shape[0]/(data3_1x.shape[0]+data3_2x.shape[0]+data3_3x.shape[0])
	priori3_2 =	data3_2x.shape[0]/(data3_1x.shape[0]+data3_2x.shape[0]+data3_3x.shape[0])
	priori3_3 =	data3_3x.shape[0]/(data3_1x.shape[0]+data3_2x.shape[0]+data3_3x.shape[0])
	priori1 = [priori1_1, priori1_2, priori1_3]
	priori2 = [priori2_1, priori2_2, priori2_3]
	priori3 = [priori3_1, priori3_2, priori3_2]

	if data_choice == 'L':
		parameters = model.compile(data1_1x,data1_2x,data1_3x,case_choice)
		results = model.test(data1_1y,data1_2y,data1_3y, parameters, priori1)
		acc = (results[0][0]+results[1][0]+results[2][0])/results[3]
		prec1 = results[0][0]/(results[0][0]+results[0][3])
		prec2 = results[1][0]/(results[1][0]+results[1][3])
		prec3 = results[2][0]/(results[2][0]+results[2][3])
		rec1 = results[0][0]/(results[0][0]+results[0][2])
		rec2 = results[1][0]/(results[1][0]+results[1][2])
		rec3 = results[2][0]/(results[2][0]+results[2][2])
		if rec1+prec2 == 0:
			fs1 = float("inf")
		else:
			fs1=2*rec1*prec1/(rec1+prec1)
		if rec2+prec2 == 0:
			fs2 = float("inf")
		else:
			fs2=2*rec2*prec2/(rec2+prec2)
		if rec3+prec3 == 0:
			fs3 = float("inf")
		else:
			fs3=2*rec3*prec3/(rec3+prec3)
		
		print('\tClass 1\tClass 2\tClass 3\n')
		print('True Positives: {}\t{}\t{}\n'.format(results[0][0],results[1][0],results[2][0]))
		print('True Negetives: {}\t{}\t{}\n'.format(results[0][1],results[1][1],results[2][1]))
		print('False Positives: {}\t{}\t{}\n'.format(results[0][3],results[1][3],results[2][3]))
		print('False Negetives: {}\t{}\t{}\n'.format(results[0][2],results[1][2],results[2][2]))
		print('Class Precision: {}\t{}\t{}\n'.format(prec1,prec2,prec3))
		print('Class Recall: {}\t{}\t{}\n'.format(rec1, rec2, rec3))
		print('Class F-Score: {}\t{}\t{}\n'.format(fs1,fs2,fs3))
		print('Classification Accuracy: {}\n'.format(acc))
		print('Mean Precision: {}\n'.format((prec1+prec2+prec3)/3))
		print('Mean Recall: {}\n'.format((rec1+rec2+rec3)/3))
		print('Mean F-Score: {}\n'.format((fs1+fs2+fs3)/3))

		plot.plot([data1_1x, data1_2x, data1_3x], [data1_1y, data1_2y, data1_3y], parameters, priori1)

	elif data_choice == 'N':
		parameters = model.compile(data2_1x,data2_2x,data2_3x,case_choice)
		results = model.test(data2_1y, data2_2y, data2_3y, parameters, priori2)
		acc = (results[0][0]+results[1][0]+results[2][0])/results[3]
		if results[0][0]+results[0][3] == 0:
			prec1 = float("inf")
		else:
			prec1 = results[0][0]/(results[0][0]+results[0][3])
		if results[1][0]+results[1][3] == 0:
			prec2 = float("inf")
		else:
			prec2 = results[1][0]/(results[1][0]+results[1][3])
		if results[2][0]+results[2][3] == 0:
			prec3 = float("inf")
		else:
			prec3 = results[2][0]/(results[2][0]+results[2][3])
		rec1 = results[0][0]/(results[0][0]+results[0][2])
		rec2 = results[1][0]/(results[1][0]+results[1][2])
		rec3 = results[2][0]/(results[2][0]+results[2][2])
		if rec1+prec2 == 0:
			fs1 = float("inf")
		else:
			fs1=2*rec1*prec1/(rec1+prec1)
		if rec2+prec2 == 0:
			fs2 = float("inf")
		else:
			fs2=2*rec2*prec2/(rec2+prec2)
		if rec3+prec3 == 0:
			fs3 = float("inf")
		else:
			fs3=2*rec3*prec3/(rec3+prec3)

		print('\tClass 1\tClass 2\tClass 3\n')
		print('True Positives: {}\t{}\t{}\n'.format(results[0][0],results[1][0],results[2][0]))
		print('True Negetives: {}\t{}\t{}\n'.format(results[0][1],results[1][1],results[2][1]))
		print('False Positives: {}\t{}\t{}\n'.format(results[0][3],results[1][3],results[2][3]))
		print('False Negetives: {}\t{}\t{}\n'.format(results[0][2],results[1][2],results[2][2]))
		print('Class Precision: {}\t{}\t{}\n'.format(prec1,prec2,prec3))
		print('Class Recall: {}\t{}\t{}\n'.format(rec1, rec2, rec3))
		print('Class F-Score: {}\t{}\t{}\n'.format(fs1,fs2,fs3))
		print('Classification Accuracy: {}\n'.format(acc))
		print('Mean Precision: {}\n'.format((prec1+prec2+prec3)/3))
		print('Mean Recall: {}\n'.format((rec1+rec2+rec3)/3))
		print('Mean F-Score: {}\n'.format((fs1+fs2+fs3)/3))

		plot.plot([data2_1x, data2_2x, data2_3x], [data2_1y, data2_2y, data2_3y], parameters, priori2)

	elif data_choice == 'R':
		parameters = model.compile(data3_1x,data3_2x,data3_3x,case_choice)
		results = model.test(data3_1y,data3_2y,data3_3y, parameters, priori3)
		acc = (results[0][0]+results[1][0]+results[2][0])/results[3]
		prec1 = results[0][0]/(results[0][0]+results[0][3])
		prec2 = results[1][0]/(results[1][0]+results[1][3])
		prec3 = results[2][0]/(results[2][0]+results[2][3])
		rec1 = results[0][0]/(results[0][0]+results[0][2])
		rec2 = results[1][0]/(results[1][0]+results[1][2])
		rec3 = results[2][0]/(results[2][0]+results[2][2])
		if rec1+prec2 == 0:
			fs1 = float("inf")
		else:
			fs1=2*rec1*prec1/(rec1+prec1)
		if rec2+prec2 == 0:
			fs2 = float("inf")
		else:
			fs2=2*rec2*prec2/(rec2+prec2)
		if rec3+prec3 == 0:
			fs3 = float("inf")
		else:
			fs3=2*rec3*prec3/(rec3+prec3)

		print('\tClass 1\tClass 2\tClass 3\n')
		print('True Positives: {}\t{}\t{}\n'.format(results[0][0],results[1][0],results[2][0]))
		print('True Negetives: {}\t{}\t{}\n'.format(results[0][1],results[1][1],results[2][1]))
		print('False Positives: {}\t{}\t{}\n'.format(results[0][3],results[1][3],results[2][3]))
		print('False Negetives: {}\t{}\t{}\n'.format(results[0][2],results[1][2],results[2][2]))
		print('Class Precision: {}\t{}\t{}\n'.format(prec1,prec2,prec3))
		print('Class Recall: {}\t{}\t{}\n'.format(rec1, rec2, rec3))
		print('Class F-Score: {}\t{}\t{}\n'.format(fs1,fs2,fs3))
		print('Classification Accuracy: {}\n'.format(acc))
		print('Mean Precision: {}\n'.format((prec1+prec2+prec3)/3))
		print('Mean Recall: {}\n'.format((rec1+rec2+rec3)/3))
		print('Mean F-Score: {}\n'.format((fs1+fs2+fs3)/3))

		plot.plot([data3_1x, data3_2x, data3_3x], [data3_1y, data3_2y, data3_3y], parameters, priori3)

	else:
		print('Wrong data choice entered.\n Exiting')
		exit(0)

if __name__ == "__main__":
	main()