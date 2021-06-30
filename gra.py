# -*- coding: utf-8 -*-
import numpy as np
X = [1,2,3,4,5,6,7,8,9]
X1 = np.array(X)
y = [1,2,3,4,5,6,7,8,9]
y1 = np.array(y) * 1
def linear_regression(X, y, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001):
     N = float(len(y))
     for i in range(epochs):
          y_current = (m_current * X) + b_current
          cost = sum([data**2 for data in (y-y_current)]) / N
          m_gradient = -(1/N) * sum(X * (y - y_current))
          b_gradient = -(1/N) * sum(y - y_current)
          m_current = m_current - (learning_rate * m_gradient)
          b_current = b_current - (learning_rate * b_gradient)
          print(m_current,b_current,cost)
     return print("New m:",m_current,"New b:", b_current,"new Cost:", cost)
linear_regression(X1, y1, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001)

