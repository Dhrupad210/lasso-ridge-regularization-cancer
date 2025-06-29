# L1 and L2 Regularization for Cancer Risk Prediction

## 🧠 Overview

This project implements **Stochastic Gradient Descent (SGD)** from scratch to train linear regression models with **L1 (Lasso)** and **L2 (Ridge)** regularization. It models the relationship between **carcinogen exposure (arbitrary units)** and **cancer risk (%)** using synthetic biological data. This educational project demonstrates how different types of regularization impact model learning and overfitting in a simplified biomedical context.

 

## 📊 Features

- 🔬 Generates **synthetic data** for 20 samples simulating carcinogen exposure and corresponding cancer risk.
- ⚙️ Implements **SGD-based linear regression** with:
  - **L1 Regularization (Lasso)** – encourages sparsity in weights.
  - **L2 Regularization (Ridge)** – penalizes large weights for stability.
- 📈 **Visualizes**:
  - Scatter plots with fitted regression lines for both L1 and L2.
  - Loss curves (MSE) over 1000 epochs.
- 📉 Reports **final model performance** using **Mean Squared Error (MSE)**.

 

## 🔧 Requirements

- Python 3.6+
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

 

## 🚀 Usage

Clone the repository and run the script:

```bash
git clone https://github.com/Dhrupad210/lasso-ridge-regularization-cancer.git
cd lasso-ridge-regularization-cancer
python sgd_regularization_cancer.py
```

This will:

- Generate and visualize synthetic cancer risk data.
- Train models using L1 and L2 regularization via SGD.
- Display:
  - Two scatter plots with regression fits.
  - One loss curve plot.
- Output final model parameters and MSE for both regularization types.

 

## 🖥️ Example Output

### Console Output:

```
L1 Regularization - Slope: 0.476, Intercept: 14.352, MSE: 55.927
L2 Regularization - Slope: 0.489, Intercept: 14.605, MSE: 55.642
```

### Plots:
- 📍 **Scatter plots** with fitted regression lines for L1 (red) and L2 (green).
- 📉 **Loss curves** showing mean squared error reduction over 1000 training epochs for both methods.

 

## 🧬 Biological Context

The synthetic dataset simulates how **exposure to carcinogens** might influence **cancer risk**. Though simplified, this project mimics real-world biological trends where environmental exposure contributes to disease risk. Regularization plays a crucial role in ensuring robust models, especially in biomedical data where features may be noisy or correlated.

 

## 📂 File Structure

```
lasso-ridge-regularization-cancer/
├── sgd_regularization_cancer.py    # Core script: data generation, training, visualization
└── README.md                       # Project documentation
```

 

## 🤝 Contributing

Pull requests are welcome! If you have suggestions to improve the code, visualization, or add real datasets — feel free to open an issue or submit a PR.

 

## ✨ Author

**Dhrupad Banerjee**  
GitHub: [@Dhrupad210](https://github.com/Dhrupad210)

 

## 📘 License

This project is open-source and available for educational and non-commercial use.
