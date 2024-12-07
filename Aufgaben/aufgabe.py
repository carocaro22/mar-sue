import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lineare_funktion(x):
    Z = np.random.normal(loc=0, scale=4.0, size=x.shape) 
    return x + Z, Z


def m_find(x_list,t_list):
    x_mean=np.mean(x_list)
    t_mean=np.mean(t_list)
    x_t_sum=0
    x_sq_sum=0

    #x_mean_list=[x_mean for i in range(len(x_list))]
    #t_mean_list=[t_mean for i in range(len(t_list))]
    #two_list=np.sum([pow(i,2) for i in (x_list-x_mean_list)])

    x_mean_list=[x_mean]*len(x_list)
    t_mean_list=[t_mean]*len(t_list)
    x_t_sum=np.sum((x_list-x_mean_list)*(t_list-t_mean_list))
    x_sq_sum=np.sum((x_list-x_mean_list)*(x_list-x_mean_list))
    
    '''
    x_t_sum=0
    for i in range(len(x_list)):
        x_temp = x_list[i]-x_mean
        t_temp = t_list[i]-t_mean
        x_t_sum += x_temp*t_temp

    
    for i in range(len(x_list)):
        x_sq_sum+=pow(x_list[i]-x_mean,2)
    '''


    return x_t_sum/x_sq_sum

def b_find(x_list,t_list, m):
    x_mean=np.mean(x_list)
    t_mean=np.mean(t_list)
    return t_mean-(m*x_mean)

def new_lineare_funktion(x,m,b):
    return x*m+b

def theta(X,Y):
    ones = np.ones(X.shape, dtype=int)
    X = np.stack((ones, X), axis=1)
    theta = np.linalg.inv(np.matmul(X.T, X))@X.T@Y
    return theta
    
def theta_mit_np(X, Y): 
    m, c = np.linalg.lstsq(X, Y)[0]
    return (m, c)

def plot(X_list, Y_list, x1, x2, y1, y2): 
    # Scatter-Plot erstellen
    plt.figure(figsize=(8, 6))
    plt.scatter(X_list, Y_list, label="X/Y", color="blue")
    plt.plot(X_list, X_list, label="line", color="red")
    plt.plot([x1,x2], [y1,y2], label="line", color="blue")
    # Beschriftungen hinzufügen
    plt.xlabel("X-Werte")
    plt.ylabel("Y-Werte")
    plt.title("Scatter Plot von X und Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    N = 5
    end_num = 100
    
    X_list = np.linspace(0, end_num, N)
    
    # Y und Z-Werte berechnen
    Y_list, Z_list = lineare_funktion(X_list)
    
    print(X_list)
    print(Z_list)
    print(Y_list)

    m = m_find(x_list=X_list,t_list=Y_list)
    b = b_find(x_list=X_list,t_list=Y_list, m=m)
    print(m)
    print(b)
    x1=0
    x2=end_num
    y1=new_lineare_funktion(x1,m,b)
    y2=new_lineare_funktion(x2,m,b)
    plot(X_list, Y_list, x1, x2, y1, y2)

    return

if __name__=="__main__":
    main()