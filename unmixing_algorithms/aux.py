#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:17:49 2024

@author: nicolasmendoza
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats as stats

def replace(arr):
    # Calculate mean and standard deviation
    mean_val = np.mean(arr)
    std_dev = np.std(arr)
    # Define a threshold for identifying outliers (e.g., 3 times the standard deviation)
    threshold = 3 * std_dev
    # Identify outliers
    outliers = np.abs(arr - mean_val) > threshold
    # Print the number of outliers
    num_outliers = np.sum(outliers) / len(arr) * 100
    print(f"% of outliers: {num_outliers}")
    # Replace outliers with the closest limit value
    arr[outliers] = np.clip(arr[outliers], mean_val - threshold, mean_val + threshold)
    return arr

def Anova(ERec, EEnd, EAb, time): 
    pd.set_option('display.float_format', '{:.2e}'.format)
    
    ############### Error de reconstruccion
    means_ERec = ERec.mean()
    std_devs_ERec = ERec.std(ddof=1)
    
    
    p_values_ERec = []
    for col in ERec.columns[1:]:
        f_statistic, p_value = stats.f_oneway(ERec['Nebeae-sc'], ERec[col])
        p_values_ERec.append(p_value)
        
    
    # Crear un gráfico de caja para visualizar los datos
    plt.figure()
    plt.boxplot(ERec, labels=['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM','UNSUBMM','SeCoDe','gtvMBO'])
    plt.title('Comparison of Methodologies')
    plt.xlabel('Metodology')
    plt.ylabel('Output Estimation Error')
    plt.show()
    
    df_ERec = pd.DataFrame({'Metodology': ['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM', 'UNSUBMM','SeCoDe','gtvMBO'],
                       'Mean': means_ERec,
                       'Standard Deviation': std_devs_ERec})
    print('Output Estimation Error')
    print(df_ERec.to_string(index=False))
    
    for i, col in enumerate(ERec.columns[1:]):
        if p_values_ERec[i] < 0.05:  # Significance level of 0.05
            print(f"Significant differences found between NEBEAE-SC and {col}.")
        else:
            print(f"No significant differences were found between NEBEAE-SC and {col}.")
    
    
    ############## EndMembers
    means_EEnd = EEnd.mean()
    std_devs_EEnd = EEnd.std(ddof=1)
    
    
    p_values_EEnd = []
    for col in EEnd.columns[1:]:
        f_statistic, p_value = stats.f_oneway(EEnd['Nebeae-sc'], EEnd[col])
        p_values_EEnd.append(p_value)
        
    
    # Crear un gráfico de caja para visualizar los datos
    plt.figure()
    plt.boxplot(EEnd, labels=['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM','UNSUBMM','SeCoDe','gtvMBO'])
    plt.title('Comparison of Methodologies')
    plt.xlabel('Metodology')
    plt.ylabel('End-members Estimation Error')
    plt.show()
    
    df_EEnd = pd.DataFrame({'Metodology': ['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM','UNSUBMM','SeCoDe','gtvMBO'],
                       'Mean': means_EEnd,
                       'Standard Deviation': std_devs_EEnd})
    print('End-members Estimation Error')
    print(df_EEnd.to_string(index=False))
    
    for i, col in enumerate(EEnd.columns[1:]):
        if p_values_EEnd[i] < 0.05:  # Significance level of 0.05
            print(f"Significant differences found between NEBEAE-SC and {col}.")
        else:
            print(f"No significant differences were found between NEBEAE-SC and {col}.")
    
    
    
    ############ Abundancias ####################
    means_EAb = np.mean(EAb, axis=0)
    std_devs_EAb = np.std(EAb, axis=0, ddof=1)
    
    p_values_EAb = []
    for col in EAb.columns[1:]:
        f_statistic, p_value = stats.f_oneway(EAb['Nebeae-sc'], EAb[col])
        p_values_EAb.append(p_value)
        
    
    # Crear un gráfico de caja para visualizar los datos
    plt.figure()
    plt.boxplot(EAb, labels=['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM','UNSUBMM','SeCoDe','gtvMBO'])
    plt.title('Comparison of Methodologies')
    plt.xlabel('Metodology')
    plt.ylabel('Abundances Estimation Error')
    plt.show()
    
    df_EAb = pd.DataFrame({'Metodology': ['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM','UNSUBMM','SeCoDe','gtvMBO'],
                       'Mean': means_EAb,
                       'Standard Deviation': std_devs_EAb})
    print('Abundances Estimation Error')
    print(df_EAb.to_string(index=False))
    
    for i, col in enumerate(EAb.columns[1:]):
        if p_values_EAb[i] < 0.05:  # Significance level of 0.05
            print(f"Significant differences found between NEBEAE-SC and {col}.")
        else:
            print(f"No significant differences were found between NEBEAE-SC and {col}.")
    

    
    ############## time ############
    means_time = time.mean()
    std_devs_time = time.std(ddof=1)
    
    p_values_time = []
    for col in time.columns[1:]:
        f_statistic, p_value = stats.f_oneway(time['Nebeae-sc'], time[col])
        p_values_time.append(p_value)
    
    # Crear un gráfico de caja para visualizar los datos
    plt.figure()
    plt.boxplot(time, labels=['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'EBEAE','UNSUBMM','SeCoDe','gtvMBO'])
    plt.title('Comparison of Methodologies')
    plt.xlabel('Metodology')
    plt.ylabel('Computational Time (s)')
    plt.show()
    
    print('Computational Time (s)')
    df_time = pd.DataFrame({'Metodology': ['NEBEAE-SC', 'NEBEAE', 'EBEAE-SC', 'G-MLM','UNSUBMM','SeCoDe','gtvMBO'],
                       'Mean': means_time,
                       'Standard Deviation': std_devs_time})
    
    print(df_time.to_string(index=False))
    
    for i, col in enumerate(time.columns[1:]):
        if p_values_time[i] < 0.05:  # Significance level of 0.05
            print(f"Significant differences found between NEBEAE-SC and {col}.")
        else:
            print(f"No significant differences were found between NEBEAE-SC and {col}.")   

def errors1(Yo, Y, Ao, A, Po, P, n):
    L, K = Y.shape
    Ep = np.zeros((n,n))
    Ea = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Ep[i,j] = np.linalg.norm(Po[:, i] - P[:, j])
            Ea[i,j]=  np.linalg.norm(Ao[i, :] - A[j, :])
            
    EY = np.linalg.norm(Yo - Y, 'fro') /np.linalg.norm(Yo,'fro')
    EP = sum(np.nanmin(Ep,axis=1)) 
    EA = sum(np.nanmin(Ea,axis=1))
    return EY, EP, EA

def color(matriz):
    k,l=matriz.shape
    A=np.zeros((k,l,3)).astype(np.uint8)
    for i in range(k):
        for j in range(l):
            p=matriz[i,j]
            if p==0:
                A[i,j,:]=[255,255,255]
            if p==1:
                A[i,j,:]=[0,255,0]
            if p==2:
                A[i,j,:]=[255,0,0]
            if p==3:
                A[i,j,:]=[0,0,255]
    return A

def maxab(A,n1,n2,n3,n4):
    n, K = A.shape
    if n != (n1 + n2 + n3 + n4):
        print('Error dimentions not match')
        return []
    abun = np.zeros((4,K))
    lim1 = range(n1)
    lim2 = range(n1,n1+n2)
    lim3 = range(n1+n2,n1+n2+n3)
    lim4 = range(n1+n2+n3,n)
    
    abun[0,:]=sum(A[lim1,:])
    abun[1,:]=sum(A[lim2,:])
    abun[2,:]=sum(A[lim3,:])
    abun[3,:]=sum(A[lim4,:])    
    eti = np.argmax(abun,axis=0,keepdims=True)
    return abun, eti

def metricas(y_true, y_pred,label):
    label_names= ['NT','TT','HT','BG']
    m = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred,display_labels=label_names);
    cm = m.confusion_matrix
    tp = np.zeros((1,4))
    fn = np.zeros((1,4))
    tn = np.zeros((1,4))
    fp = np.zeros((1,4))
    for i in range(4):
        tp[0,i] = cm[i,i]
        fn[0,i] = sum(cm[i,:])- cm[i,i]
        fp[0,i] = sum(cm[:,i])- cm[i,i]
        tn[0,i]= cm.sum()-(tp[0,i] + fn[0,i] + fp[0,i])
    acc = ((tp+tn)/(tp+tn+fn+fp)).ravel()
    sens =( tp/(tp+fn)).ravel()
    spec = (tn/(tn+fp)).ravel()
    f1 = (2*tp/(2*tp+fp)).ravel() 
    met = pd.DataFrame({'Tissue' :['NT','TT','HT','BG'],'accuracy':acc,'sensivility':sens,'specifity':spec,'f1_score':f1})
    met.set_index('Tissue',inplace=True)
    met=met.T
    #print('metricas por clase: \n',met)
    #print(label)
    print('exactitud general =', np.sum(np.diag(cm))/np.sum(cm))
    return cm,met

def metrics2(y_true, y_pred, label_names):
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    fp = np.sum(cm, axis=0) - tp
    tn = np.sum(cm) - (tp + fn + fp)

    acc = ((tp + tn) / (tp + tn + fn + fp)).ravel()
    sens = (tp / (tp + fn)).ravel()
    spec = (tn / (tn + fp)).ravel()
    f1 = (2 * tp / (2 * tp + fp)).ravel()

    met = pd.DataFrame({
        'Tissue': label_names,
        'accuracy': acc,
        'sensivility': sens,
        'specifity': spec,
        'f1_score': f1
    })
    met.set_index('Tissue', inplace=True)
    met = met.T

    print('exactitud general =', np.sum(np.diag(cm)) / np.sum(cm))
    
    return cm, met

def errorabundances(Ao, A):
    """
    Translates the MATLAB function 'errorabundances' to Python.

    Args:
        Ao: A NumPy array representing the original data.
        A: A NumPy array representing the data with potential errors.

    Returns:
        Ea: The average error abundance.
    """

    N, K = Ao.shape  # Get dimensions of Ao
    Eaa = np.zeros(N)

    for i in range(N):
        Aoi = Ao[i, :]
        Error = np.linalg.norm(Aoi - A, axis=1) / np.linalg.norm(Aoi)
        Eaa[i] = np.min(Error)

    Ea = np.sum(Eaa) / N

    return Ea
def errorendmembers(Po,P):
    """
    Translates the MATLAB function 'errorendmembers' to Python.

    Args:
        Po: A NumPy array representing the original endmembers.
        P: A NumPy array representing the estimated endmembers.

    Returns:
        Ep: The average error of the endmembers.
    """

    N = Po.shape[1]  # Get the number of endmembers
    Epp = np.zeros(N)
    P = P/ np.sum(P,axis=0,keepdims=True)

    for i in range(N):
        Poi = Po[:, i] / np.sum(Po[:, i])  # Normalize Po[:, i]
        Error = np.linalg.norm(Poi[:, np.newaxis] - P, axis=0) / np.linalg.norm(Poi)
        Epp[i] = np.min(Error)

    Ep = np.sum(Epp) / N

    return Ep
def errorSAM(Po, P):
    """
    Computes the average Spectral Angle Mapper (SAM) error between original and estimated endmembers.

    Args:
        Po: A NumPy array representing the original endmembers.
        P: A NumPy array representing the estimated endmembers.

    Returns:
        Ep: The average SAM error.
    """

    # Normalize columns to sum 1
    Po_norm = Po / np.sum(Po, axis=0, keepdims=True)
    P_norm = P / np.sum(P, axis=0, keepdims=True)

    # Compute cosine similarity matrix
    dot_products = Po_norm.T @ P_norm
    norms = np.linalg.norm(Po_norm, axis=0)[:, None] * np.linalg.norm(P_norm, axis=0)
    cosine_sim = np.clip(dot_products / norms, -1, 1)

    # Compute SAM errors and find minimum error for each endmember
    Errors = np.arccos(cosine_sim)
    Epp = np.min(Errors, axis=1)

    # Compute the average error
    Ep = np.mean(Epp)

    return Ep