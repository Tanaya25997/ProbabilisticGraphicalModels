import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def calculate_parameters(X_test,RV,pa_index,val, trn):

    


    len_Data = len(X_test)
    theta = {}
    
    avg_log_likelihood = 0

    for rv in range(len(RV)):

        if len(pa_index[rv]) == 0:
            for v in val[rv]:
                count = np.count_nonzero(X_test == v, axis = 0)[rv]
                if trn == 1:
                    x = RV[rv]+str(v)
                    theta[x] = {'Count':count,'LL':count/len_Data}              
                    avg_log_likelihood = avg_log_likelihood + (count*np.log(count/len_Data))/len_Data
                else:
                    x = RV[rv]+str(v)
                    theta[x] = {'Count':count} 
                    avg_log_likelihood = 0

        else:
            
            comb_array = [[]]           
            arrays = []
            for i in pa_index[rv]:
                arrays.append(np.array(val[i]))
            comb_array = np.array(np.meshgrid(*arrays)).T.reshape(-1, len(pa_index[rv]))
            
            
            for v in val[rv]:
                for combination in comb_array:  
                    final_rows = []
                    comb_rows = []                 
                    arg = [X_test[:, pa_index[rv][i]] == combination[i] for i in range(len(combination))]
                                      
                    if len(pa_index[rv]) == 1:
                        combination_rows_count = np.count_nonzero(X_test == combination[0], axis = 0)[pa_index[rv][0]]
                        final_rows = np.logical_and(*arg, X_test[:, rv] == v)
                        final_rows_count = np.count_nonzero(final_rows == True,axis = 0)
                    else:
                        comb_rows = np.logical_and(*arg)
                        combination_rows_count = np.count_nonzero(comb_rows == True,axis = 0)
                        final_rows = X_test[comb_rows, rv] == v
                        final_rows_count = np.sum(final_rows)

 
                    if trn == 1: 
                        y = ''
                        for i in range(len(combination)):
                            y = y+RV[pa_index[rv][i]]+str(combination[i])
                        x =  RV[rv]+str(v)+y
                        theta[x] = {'Count': final_rows_count, 'LL': final_rows_count/combination_rows_count} 
                        if (combination_rows_count!=0) & (final_rows_count!=0):
                            avg_log_likelihood = avg_log_likelihood + (final_rows_count*np.log(final_rows_count/combination_rows_count))/len_Data
                        else:
                            avg_log_likelihood = avg_log_likelihood + 0.00001
                    else:
                        y = ''
                        for i in range(len(combination)):
                            y = y+RV[pa_index[rv][i]]+str(combination[i])
                        x =  RV[rv]+str(v)+y
                        theta[x] =  {'Count': final_rows_count} 
                        avg_log_likelihood = 0
                        
    
    return theta, np.round(avg_log_likelihood,4)


def calculate_avg_loglikelihood_test(theta,count,X_test_length):

    avg_likelihood_testdata = 0
    
    for i in theta:
        if theta.get(i)['LL'] != 0:
            avg_likelihood_testdata = avg_likelihood_testdata + (count.get(i)['Count']*np.log(theta.get(i)['LL']))/X_test_length
        else:
            avg_likelihood_testdata = avg_likelihood_testdata + 0.00001
     
    return np.round(avg_likelihood_testdata,4)
            

def predict(Xtest,total,theta):

    
    correct_predict = 0
    for i in range(total):
        
        D1 = theta.get('D1H'+str(Xtest[i][4])+'B'+str(Xtest[i][3]))['LL']*theta.get('A'+str(Xtest[i][0]))['LL']*theta.get('G'+str(Xtest[i][1]))['LL']*theta.get('C'+str(Xtest[i][2])+'D1')['LL']*theta.get('B'+str(Xtest[i][3])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('H'+str(Xtest[i][4])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('E'+str(Xtest[i][5])+'D1')['LL']*theta.get('R'+str(Xtest[i][6])+'A'+str(+Xtest[i][0])+'D1')['LL']*theta.get('I'+str(Xtest[i][7])+'D1')['LL']
        D2 = theta.get('D2H'+str(Xtest[i][4])+'B'+str(Xtest[i][3]))['LL']*theta.get('A'+str(Xtest[i][0]))['LL']*theta.get('G'+str(Xtest[i][1]))['LL']*theta.get('C'+str(Xtest[i][2])+'D2')['LL']*theta.get('B'+str(Xtest[i][3])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('H'+str(Xtest[i][4])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('E'+str(Xtest[i][5])+'D2')['LL']*theta.get('R'+str(Xtest[i][6])+'A'+str(Xtest[i][0])+'D2')['LL']*theta.get('I'+str(Xtest[i][7])+'D2')['LL']
        pD1 = D1/(D1+D2)
        pD2 = D2/(D1+D2)
        if pD1 > pD2:
            pred = 1
        else:
            pred = 2
        if pred == Xtest[i][8]:
            correct_predict+=1
            
    prediction_accuracy = np.round(correct_predict/total,4)    
        
    return prediction_accuracy


def predict_new(Xtest,total,theta):


    
    correct_predict = 0
    for i in range(total):
        
        D1 = theta.get('D1H'+str(Xtest[i][4])+'B'+str(Xtest[i][3]))['LL']*theta.get('A'+str(Xtest[i][0]))['LL']*theta.get('G'+str(Xtest[i][1]))['LL']*theta.get('C'+str(Xtest[i][2])+'A'+str(+Xtest[i][0])+'D1')['LL']*theta.get('B'+str(Xtest[i][3])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('H'+str(Xtest[i][4])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('E'+str(Xtest[i][5])+'D1')['LL']*theta.get('R'+str(Xtest[i][6])+'A'+str(+Xtest[i][0])+'D1')['LL']*theta.get('I'+str(Xtest[i][7])+'A'+str(+Xtest[i][0])+'D1')['LL']
        D2 = theta.get('D2H'+str(Xtest[i][4])+'B'+str(Xtest[i][3]))['LL']*theta.get('A'+str(Xtest[i][0]))['LL']*theta.get('G'+str(Xtest[i][1]))['LL']*theta.get('C'+str(Xtest[i][2])+'A'+str(+Xtest[i][0])+'D2')['LL']*theta.get('B'+str(Xtest[i][3])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('H'+str(Xtest[i][4])+'A'+str(Xtest[i][0])+'G'+str(Xtest[i][1]))['LL']*theta.get('E'+str(Xtest[i][5])+'D2')['LL']*theta.get('R'+str(Xtest[i][6])+'A'+str(Xtest[i][0])+'D2')['LL']*theta.get('I'+str(Xtest[i][7])+'A'+str(+Xtest[i][0])+'D2')['LL']
        pD1 = D1/(D1+D2)
        pD2 = D2/(D1+D2)
        if pD1 > pD2:
            pred = 1
        else:
            pred = 2
        if pred == Xtest[i][8]:
            correct_predict+=1
            
    prediction_accuracy = np.round(correct_predict/total,4)    
        
    return prediction_accuracy

if __name__ == '__main__':

    
       
    RV = ['A','G','C','B','H','E','R','I','D']   
    pa_index = [[],[],[8],[0,1],[0,1],[8],[0,8],[8],[4,3]]  
    val = [[1,2,3],[1,2],[1,2,3,4],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]]
    
    X_train = np.genfromtxt("../data/data-train-1.txt", dtype=str, delimiter=",")
    X_train = np.array(X_train, dtype='int')
    
    ##### Question 5#####
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("\n Question 5: Implement maximum likelihood learning for all factors in the directed model.For this question, use the data in data-train-1.txt only. What is the average log-likelihood over all the datapoints?")
    print("--------------------------------------------------------------------------------")
    theta, avg_log_likelihood = calculate_parameters(X_train,RV,pa_index,val,1)
    print("\n", theta)
    print("\n Average log likelihood of data-train-1.txt is: ", avg_log_likelihood)
    
    
    
    ##### Question 7 #####
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("\n Question 7: Probability Queries")
    print("--------------------------------------------------------------------------------")
    print("\n Probability that P(E = 1|A = 2,G = 1,C = 1,B = 1,H = 2,R = 2, I = 1,D = 1) is ",  np.round(theta.get('E1D1')['LL'],4))
    print("\n Probability that P(B = 1|A = 3,G = 1,C = 3,H = 1,E = 2,R = 1, I = 1,D = 2) is ", np.round((theta.get('B1A3G1')['LL']*theta.get('D2H1B1')['LL'])/(theta.get('B1A3G1')['LL']*theta.get('D2H1B1')['LL'] + theta.get('B2A3G1')['LL']*theta.get('D2H1B2')['LL']),4))
    print("\n Probability that P(A = 1,H = 1|G = 2,C = 3,B = 2,E = 1,R = 2, I = 1,D = 1) is ", np.round((theta.get('A1')['LL']*theta.get('H1A1G2')['LL']*theta.get('B2A1G2')['LL']*theta.get('R2A1D1')['LL']*theta.get('D1H1B2')['LL'])/(theta.get('A1')['LL']*theta.get('H1A1G2')['LL']*theta.get('B2A1G2')['LL']*theta.get('R2A1D1')['LL']*theta.get('D1H1B2')['LL'] + theta.get('A1')['LL']*theta.get('H2A1G2')['LL']*theta.get('B2A1G2')['LL']*theta.get('R2A1D1')['LL']*theta.get('D1H2B2')['LL'] + theta.get('A2')['LL']*theta.get('H1A2G2')['LL']*theta.get('B2A2G2')['LL']*theta.get('R2A2D1')['LL']*theta.get('D1H1B2')['LL'] + theta.get('A2')['LL']*theta.get('H2A2G2')['LL']*theta.get('B2A2G2')['LL']*theta.get('R2A2D1')['LL']*theta.get('D1H2B2')['LL'] + theta.get('A3')['LL']*theta.get('H1A3G2')['LL']*theta.get('B2A3G2')['LL']*theta.get('R2A3D1')['LL']*theta.get('D1H1B2')['LL'] + theta.get('A3')['LL']*theta.get('H2A3G2')['LL']*theta.get('B2A3G2')['LL']*theta.get('R2A3D1')['LL']*theta.get('D1H2B2')['LL']),4))
    print("\n Probability that P(D = 2|A = 1,G = 2,C = 1,B = 1,H = 1,E = 2, I = 2) is ", np.round((theta.get('D2H1B1')['LL']*theta.get('C1D2')['LL']*theta.get('E2D2')['LL']*theta.get('I2D2')['LL'])/(theta.get('D1H1B1')['LL']*theta.get('C1D1')['LL']*theta.get('E2D1')['LL']*theta.get('I2D1')['LL'] + theta.get('D2H1B1')['LL']*theta.get('C1D2')['LL']*theta.get('E2D2')['LL']*theta.get('I2D2')['LL']),4))
    
    ##### Question 8 #####
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("\n Question 8: Test Log-likelihood")
    print("--------------------------------------------------------------------------------")
    X_train_1 = np.genfromtxt("../data/data-train-1.txt", dtype=str, delimiter=",")
    X_train_1 = np.array(X_train_1, dtype='int')
    X_train_2 = np.genfromtxt("../data/data-train-2.txt", dtype=str, delimiter=",")
    X_train_2 = np.array(X_train_2, dtype='int')
    X_train_3 = np.genfromtxt("../data/data-train-3.txt", dtype=str, delimiter=",")
    X_train_3 = np.array(X_train_3, dtype='int')
    X_train_4 = np.genfromtxt("../data/data-train-4.txt", dtype=str, delimiter=",")
    X_train_4 = np.array(X_train_4, dtype='int')
    X_train_5 = np.genfromtxt("../data/data-train-5.txt", dtype=str, delimiter=",")
    X_train_5 = np.array(X_train_5, dtype='int')
    
    theta_1,avg_loglikelihood_1 = calculate_parameters(X_train_1,RV,pa_index,val,1)
    theta_2,avg_loglikelihood_2 = calculate_parameters(X_train_2,RV,pa_index,val,1)
    theta_3,avg_loglikelihood_3 = calculate_parameters(X_train_3,RV,pa_index,val,1)
    theta_4,avg_loglikelihood_4 = calculate_parameters(X_train_4,RV,pa_index,val,1)
    theta_5,avg_loglikelihood_5 = calculate_parameters(X_train_5,RV,pa_index,val,1)
    
    print("\nAverage log liklihood for training set is: ", avg_loglikelihood_1, avg_loglikelihood_2, avg_loglikelihood_3, avg_loglikelihood_4, avg_loglikelihood_5)
    
 
    X_test_1 = np.genfromtxt("../data/data-test-1.txt", dtype=str, delimiter=",")
    X_test_1 = np.array(X_test_1, dtype='int')
    X_test_1_length = len(X_test_1)
    X_test_2 = np.genfromtxt("../data/data-test-2.txt", dtype=str, delimiter=",")
    X_test_2 = np.array(X_test_2, dtype='int')
    X_test_2_length = len(X_test_2)
    X_test_3 = np.genfromtxt("../data/data-test-3.txt", dtype=str, delimiter=",")
    X_test_3 = np.array(X_test_3, dtype='int')
    X_test_3_length = len(X_test_3)
    X_test_4 = np.genfromtxt("../data/data-test-4.txt", dtype=str, delimiter=",")
    X_test_4 = np.array(X_test_4, dtype='int')
    X_test_4_length = len(X_test_4)
    X_test_5 = np.genfromtxt("../data/data-test-5.txt", dtype=str, delimiter=",")
    X_test_5 = np.array(X_test_5, dtype='int')
    X_test_5_length = len(X_test_5)
     
    count_1,_1 = calculate_parameters(X_test_1,RV,pa_index,val,0)
    count_2,_2 = calculate_parameters(X_test_2,RV,pa_index,val,0)
    count_3,_3 = calculate_parameters(X_test_3,RV,pa_index,val,0)
    count_4,_4 = calculate_parameters(X_test_4,RV,pa_index,val,0)
    count_5,_5 = calculate_parameters(X_test_5,RV,pa_index,val,0)
     
    avg_loglikelihood_test_1 = calculate_avg_loglikelihood_test(theta_1,count_1,X_test_1_length)
    avg_loglikelihood_test_2 = calculate_avg_loglikelihood_test(theta_2,count_2,X_test_2_length)
    avg_loglikelihood_test_3 = calculate_avg_loglikelihood_test(theta_3,count_3,X_test_3_length)
    avg_loglikelihood_test_4 = calculate_avg_loglikelihood_test(theta_4,count_4,X_test_4_length)
    avg_loglikelihood_test_5 = calculate_avg_loglikelihood_test(theta_5,count_5,X_test_5_length)
    
            
    print("Average log liklihood for testset is: ", avg_loglikelihood_test_1, avg_loglikelihood_test_2, avg_loglikelihood_test_3, avg_loglikelihood_test_4, avg_loglikelihood_test_5)
    
    
    print("\nMean of train log likelihoods is = ",np.round(np.average([avg_loglikelihood_1, avg_loglikelihood_2, avg_loglikelihood_3, avg_loglikelihood_4, avg_loglikelihood_5]),4))
    
    print("Mean of test log likelihoods is = ",np.round(np.average([avg_loglikelihood_test_1, avg_loglikelihood_test_2, avg_loglikelihood_test_3, avg_loglikelihood_test_4, avg_loglikelihood_test_5]),4))

    print("\nStandard deviation of train data log likelihoods is: ", np.round(np.std([avg_loglikelihood_1, avg_loglikelihood_2, avg_loglikelihood_3, avg_loglikelihood_4, avg_loglikelihood_5]),4))
    print("Standard deviation of test data log likelihoods is: ", np.round(np.std([avg_loglikelihood_test_1, avg_loglikelihood_test_2, avg_loglikelihood_test_3, avg_loglikelihood_test_4, avg_loglikelihood_test_5]),4))

    #### QUESTION 9 ######
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("\nQuestion 9: Prediction Accuracy for Test files")
    print("--------------------------------------------------------------------------------")
    
    acc_1 = predict(X_test_1,X_test_1_length,theta_1)  
    print("\nPrediction accuracy for data-test-1.txt is : ", acc_1)
    
    acc_2 = predict(X_test_2,X_test_2_length,theta_2)
    print("Prediction accuracy for data-test-2.txt is : ", acc_2)
    
    acc_3 = predict(X_test_3,X_test_3_length,theta_3)
    print("Prediction accuracy for data-test-3.txt is : ", acc_3)
    acc_4 = predict(X_test_4,X_test_4_length,theta_4)
    print("Prediction accuracy for data-test-4.txt is : ", acc_4)
    acc_5 = predict(X_test_5,X_test_5_length,theta_5)
    print("Prediction accuracy for data-test-5.txt is : ", acc_5)
    print("Mean of prediction accuracy over the five datasets is = ",np.round(np.average([acc_1,acc_2,acc_3,acc_4,acc_5]),4))
    print("Standard Deviation of prediction accuracy over the five datasets is = ",np.round(np.std([acc_1,acc_2,acc_3,acc_4,acc_5]),4))
    
    
    #### QUESTION 10 ######
    print("\n\n\n")
    print("--------------------------------------------------------------------------------")
    print("\nModeling: Use your own intuition about heart disease to design your own network structure for the heart disease domain.")
    print("--------------------------------------------------------------------------------")
    RV_new = ['A','G','C','B','H','E','R','I','D']   
    pa_index_new = [[],[],[0,8],[0,1],[0,1],[8],[0,8],[0,8],[4,3]]  
    
    theta_1_new,avg_loglikelihood_1_new = calculate_parameters(X_train_1,RV_new,pa_index_new,val,1)
    theta_2_new,avg_loglikelihood_2_new = calculate_parameters(X_train_2,RV_new,pa_index_new,val,1)
    theta_3_new,avg_loglikelihood_3_new = calculate_parameters(X_train_3,RV_new,pa_index_new,val,1)
    theta_4_new,avg_loglikelihood_4_new = calculate_parameters(X_train_4,RV_new,pa_index_new,val,1)
    theta_5_new,avg_loglikelihood_5_new = calculate_parameters(X_train_5,RV_new,pa_index_new,val,1)
    
    print("\nAverage log liklihood for training set is: ", avg_loglikelihood_1_new, avg_loglikelihood_2_new, avg_loglikelihood_3_new, avg_loglikelihood_4_new, avg_loglikelihood_5_new)
    
    count_1_new,_1_new = calculate_parameters(X_test_1,RV_new,pa_index_new,val,0)
    count_2_new,_2_new = calculate_parameters(X_test_2,RV_new,pa_index_new,val,0)
    count_3_new,_3_new = calculate_parameters(X_test_3,RV_new,pa_index_new,val,0)
    count_4_new,_4_new = calculate_parameters(X_test_4,RV_new,pa_index_new,val,0)
    count_5_new,_5_new = calculate_parameters(X_test_5,RV_new,pa_index_new,val,0)
    
    avg_loglikelihood_test_1_new = calculate_avg_loglikelihood_test(theta_1_new,count_1_new,X_test_1_length)
    avg_loglikelihood_test_2_new = calculate_avg_loglikelihood_test(theta_2_new,count_2_new,X_test_2_length)
    avg_loglikelihood_test_3_new = calculate_avg_loglikelihood_test(theta_3_new,count_3_new,X_test_3_length)
    avg_loglikelihood_test_4_new = calculate_avg_loglikelihood_test(theta_4_new,count_4_new,X_test_4_length)
    avg_loglikelihood_test_5_new = calculate_avg_loglikelihood_test(theta_5_new,count_5_new,X_test_5_length)
    
            
    print("Average log liklihood for testset is: ", avg_loglikelihood_test_1_new, avg_loglikelihood_test_2_new, avg_loglikelihood_test_3_new, avg_loglikelihood_test_4_new, avg_loglikelihood_test_5_new)
    
    
    print("\nMean of train log likelihoods is = ",np.round(np.average([avg_loglikelihood_1_new, avg_loglikelihood_2_new, avg_loglikelihood_3_new, avg_loglikelihood_4_new, avg_loglikelihood_5_new]),4))
    
    print("Mean of test log likelihoods is = ",np.round(np.average([avg_loglikelihood_test_1_new, avg_loglikelihood_test_2_new, avg_loglikelihood_test_3_new, avg_loglikelihood_test_4_new, avg_loglikelihood_test_5_new]),4))

    print("\nStandard deviation of train data log likelihoods is: ", np.round(np.std([avg_loglikelihood_1_new, avg_loglikelihood_2_new, avg_loglikelihood_3_new, avg_loglikelihood_4_new, avg_loglikelihood_5_new]),4))
    print("Standard deviation of test data log likelihoods is: ", np.round(np.std([avg_loglikelihood_test_1_new, avg_loglikelihood_test_2_new, avg_loglikelihood_test_3_new, avg_loglikelihood_test_4_new, avg_loglikelihood_test_5_new]),4))

    acc_1_new = predict_new(X_test_1,X_test_1_length,theta_1_new)  
    print("\nPrediction accuracy for data-test-1.txt is : ", acc_1_new) 
    acc_2_new = predict_new(X_test_2,X_test_2_length,theta_2_new)
    print("Prediction accuracy for data-test-2.txt is : ", acc_2_new)  
    acc_3_new = predict_new(X_test_3,X_test_3_length,theta_3_new)
    print("Prediction accuracy for data-test-3.txt is : ", acc_3_new)
    acc_4_new = predict_new(X_test_4,X_test_4_length,theta_4_new)
    print("Prediction accuracy for data-test-4.txt is : ", acc_4_new)
    acc_5_new = predict_new(X_test_5,X_test_5_length,theta_5_new)
    print("Prediction accuracy for data-test-5.txt is : ", acc_5_new)
    print("Mean of prediction accuracy over the five datasets is = ",np.round(np.average([acc_1_new,acc_2_new,acc_3_new,acc_4_new,acc_5_new]),4))
    print("Standard Deviation of prediction accuracy over the five datasets is = ",np.round(np.std([acc_1_new,acc_2_new,acc_3_new,acc_4_new,acc_5_new]),4))
