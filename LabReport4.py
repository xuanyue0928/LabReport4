import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="NN Activation Functions", layout="wide")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Dataset")
    ds = st.selectbox("Select toy dataset",
                      ["Linear (2 classes)", "Moons (nonlinear)", "Circles (nonlinear)"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)

    st.header("Model")
    model_type = st.selectbox("Model", ["Perceptron (linear)", "MLP (nonlinear)"])
    
    # Dropdown for the 3 required activation functions
    activation_display = st.selectbox("MLP activation", ["relu", "sigmoid", "tanh"])
    
    hidden = st.text_input("Hidden layer sizes (e.g., 16,16)", value="16,16")
    alpha = st.slider("L2 regularization (alpha)", 0.0001, 0.01, 0.0005, 0.0001)
    max_iter = st.slider("Max Iterations/Epochs", 100, 3000, 800, 100)
    random_state = st.number_input("Random seed", 0, 9999, 42)

# --- Dynamic Main Title ---
# Replaces the static title from the original foundations app
st.title(f"{activation_display.capitalize()} Activation Function")
st.write("Explore linear vs. non-linear decision boundaries and activation effects.")

def make_linear(n, noise, rs):
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, class_sep=2.0, random_state=rs)
    if noise > 0:
        rng = np.random.RandomState(rs)
        X += noise * rng.randn(*X.shape)
    return X, y

# --- Data ---
if ds == "Linear (2 classes)":
    X, y = make_linear(n_samples, noise, random_state)
elif ds == "Moons (nonlinear)":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)

# --- Model ---
if model_type.startswith("Perceptron"):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", Perceptron(max_iter=max_iter, random_state=random_state))
    ])
else:
    hidden_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip().isdigit())
    if not hidden_sizes:
        hidden_sizes = (16, 16)
        
    # Map 'sigmoid' back to 'logistic' for scikit-learn compatibility
    sklearn_activation = "logistic" if activation_display == "sigmoid" else activation_display
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                              activation=sklearn_activation,
                              alpha=alpha,
                              learning_rate_init=0.01,
                              max_iter=max_iter,
                              random_state=random_state))
    ])

# --- Fit ---
clf.fit(X, y)
score = clf.score(X, y)

# --- Side-by-Side Layout Setup ---
col1, col2 = st.columns(2)

# --- Left Column: Decision Boundary ---
with col1:
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap_light = ListedColormap(["#FFEEEE", "#EEFFEE"])
    cmap_bold = ["#CC0000", "#007700"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
    ax.scatter(X[:,0], X[:,1], c=y, s=25, edgecolor="k", cmap=ListedColormap(cmap_bold))
    ax.set_title(f"{model_type} Decision Boundary (acc={score:.3f})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    st.pyplot(fig)

# --- Right Column: Activation Function Viewer ---
with col2:
    xgrid = np.linspace(-6, 6, 400)

    def act_curve(name, x):
        if name == "relu":
            return np.maximum(0, x)
        if name == "tanh":
            return np.tanh(x)
        if name == "sigmoid": 
            return 1/(1+np.exp(-x))
        return x

    fig2, ax2 = plt.subplots(figsize=(6, 5))

    # Plot only the function currently selected in the sidebar
    y_vals = act_curve(activation_display, xgrid)
    ax2.plot(xgrid, y_vals, label=activation_display.capitalize(), color="blue", linewidth=2)

    ax2.legend()
    ax2.grid(True)
    ax2.set_title(f"Mathematical Curve for {activation_display.capitalize()}")
    ax2.set_xlabel("Input (x)")
    ax2.set_ylabel("Output (y)")
    st.pyplot(fig2)