import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y = np.array([0,0,0,1,0,1,1,1])

print(X)

model = LogisticRegression()
model.fit(X,y)

test = np.array([[0]])
ans = model.predict_proba(test)
print("will i fail? ",ans)