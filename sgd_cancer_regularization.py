import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: 20 samples of carcinogen exposure vs. cancer risk (%)
np.random.seed(42)
carcinogen_exposure=np.random.uniform(10,100,20)
cancer_risk=0.5*carcinogen_exposure+np.random.normal(0,8,20)+15

# SGD with L1 and L2 regularization
def sgd_regularized(X,y,reg_type='L2',lambda_reg=0.01,lr=0.0001,epochs=1000):
    m,b,n=0.0,0.0,len(X)
    losses=[]
    for epoch in range(epochs):
        indices=np.random.permutation(n)
        X_shuffled,y_shuffled=X[indices],y[indices]
        total_loss=0
        for i in range(n):
            xi,yi=X_shuffled[i],y_shuffled[i]
            y_pred=m*xi+b
            dm=-2*xi*(yi-y_pred)
            db=-2*(yi-y_pred)
            if reg_type=='L1':
                dm+=lambda_reg*np.sign(m)
            elif reg_type=='L2':
                dm+=2*lambda_reg*m
            m-=lr*dm
            b-=lr*db
            loss=(yi-y_pred)**2
            if reg_type=='L1':
                loss+=lambda_reg*abs(m)
            elif reg_type=='L2':
                loss+=lambda_reg*m**2
            total_loss+=loss
        losses.append(total_loss/n)
    return m,b,losses

# Run SGD for L1 and L2 regularization
slope_l1,intercept_l1,losses_l1=sgd_regularized(carcinogen_exposure,cancer_risk,'L1')
slope_l2,intercept_l2,losses_l2=sgd_regularized(carcinogen_exposure,cancer_risk,'L2')

# Calculate MSE for final models
y_pred_l1=slope_l1*carcinogen_exposure+intercept_l1
y_pred_l2=slope_l2*carcinogen_exposure+intercept_l2
mse_l1=np.mean((cancer_risk-y_pred_l1)**2)
mse_l2=np.mean((cancer_risk-y_pred_l2)**2)

# Print results
print(f"L1 Regularization - Slope: {slope_l1:.3f}, Intercept: {intercept_l1:.3f}, MSE: {mse_l1:.3f}")
print(f"L2 Regularization - Slope: {slope_l2:.3f}, Intercept: {intercept_l2:.3f}, MSE: {mse_l2:.3f}")

# Plot data and fitted lines
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(carcinogen_exposure,cancer_risk,color='blue',label='Data')
plt.plot(carcinogen_exposure,slope_l1*carcinogen_exposure+intercept_l1,color='red',label='L1 Fit')
plt.xlabel('Carcinogen Exposure')
plt.ylabel('Cancer Risk (%)')
plt.title('L1 Regularization')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(carcinogen_exposure,cancer_risk,color='blue',label='Data')
plt.plot(carcinogen_exposure,slope_l2*carcinogen_exposure+intercept_l2,color='green',label='L2 Fit')
plt.xlabel('Carcinogen Exposure')
plt.ylabel('Cancer Risk (%)')
plt.title('L2 Regularization')
plt.legend()
plt.tight_layout()
plt.show()

# Plot loss over epochs
plt.figure(figsize=(10,5))
plt.plot(losses_l1,color='red',label='L1 Loss')
plt.plot(losses_l2,color='green',label='L2 Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Loss Over Epochs')
plt.legend()
plt.show()
