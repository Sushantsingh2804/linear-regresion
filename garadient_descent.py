import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

x_train = df['x']
y_train = df['y']

lamb = 0.01
m = 58
theta0 = 0
theta1 = 0
temp0 = 0
temp1 = 0
alpha = 0.01
n = 0
hypothesis = 0
error_test = []
j = 1
while n < 2000 and j >= 0.00000000000000000000001:
    i = 0
    sum1 = 0
    sum2 = 0
    while i < m:
        hypothesis = theta0+(theta1*x_train[i])
        sum1 = sum1+(hypothesis-y_train[i])
        sum2 = sum2+((hypothesis-y_train[i])*x_train[i])
        i = i+1

    j = ((sum1*sum1)-(lamb*(theta1*theta1)))/(2*m)
    temp0 = theta0-(alpha*(sum1/m))
    temp1 = theta1-(alpha*((sum2+(lamb*theta1))/m))
    theta0 = temp0
    theta1 = temp1
    error_test.append(j)
    print("Training set:-",n)
    n = n+1

plt.plot(error_test, marker='.', color='blue')
plt.xlabel('number of iterations')
plt.ylabel('cost function or square error function')
plt.title('Graph to check gradient decent ')
plt.show()

def graph (z):
    plt.scatter(x_train, y_train,marker='x',color='red',s=20,label='dataset')
    plt.plot(x_train, z, alpha=0.5,label='hypothesis')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.xlim(4 , 24)
    plt.ylim(-5 , 25)
    plt.title('Scatter plot of training data')
    plt.legend()
    plt.show()
    return

def test (theta0,theta1):
    hypo = []
    for i in x_train:
        h = theta0 + (theta1 * i)
        hypo.append(h)

    graph(hypo)


test(theta0,theta1)

