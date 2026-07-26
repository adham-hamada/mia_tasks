import numpy as np
import matplotlib.pyplot as plt

from fifa_common import load_fifa_data, clean_and_select, get_feature_matrix, standardize, ALL_FEATURES

# Data load and preprocessing
def load_and_preprocess():
    raw = load_fifa_data()
    df = clean_and_select(raw)          # drops rows with missing Position/skill data
    X = get_feature_matrix(df)          # shape: (n_players, 9)  -> 8 skills + Value
    positions = df["Position"].to_numpy()
    names = df["Name"].to_numpy()
    print(f"[pca] feature matrix shape: {X.shape}")
    return df, X, positions, names

def covariance_matrix(Z):
    """
    Compute the covariance matrix of the standardized data by hand:
        cov = (Z^T . Z) / (n - 1)
    Z is (n_samples, n_features), so Z^T . Z is (n_features, n_features) —
    exactly the shape a covariance matrix should be. Because Z is already
    standardized (mean 0, std 1 per column), this is equivalent to the
    correlation matrix of the original data.
    """
    n = Z.shape[0]
    return (Z.T @ Z) / (n - 1)

def eigen_decompose(cov):
    """
    numpy.linalg.eig is explicitly permitted by the task for this step.
    Returns eigenvalues (1D array) and eigenvectors (columns of a 2D array).
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # The covariance matrix is symmetric, so eigenvalues are mathematically
    # guaranteed to be real — but np.linalg.eig always returns a complex
    # dtype as a safeguard for the general case. Drop the (zero) imaginary
    # part so downstream sorting/plotting works with plain floats.
    return eigenvalues.real, eigenvectors.real

def sort_components(eigenvalues, eigenvectors):
    """Sort eigenvalues descending, and reorder eigenvector columns to match."""
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]

def project_data(Z, eigenvectors, n_components):
    """
    Build a projection matrix from the top `n_components` eigenvectors and
    multiply the standardized data by it to get the low-dimensional
    representation: (n_samples, n_features) @ (n_features, n_components)
    -> (n_samples, n_components)
    """
    projection_matrix = eigenvectors[:, :n_components]
    return Z @ projection_matrix

# visualization

def plot_scree(eigenvalues):
    explained = eigenvalues / eigenvalues.sum() * 100
    cumulative = np.cumsum(explained)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(eigenvalues) + 1)
    ax1.bar(x, explained, color="#c1121f")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained variance (%)", color="#c1121f")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x])

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, color="#212529", marker="o")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_ylim(0, 105)

    plt.title("Scree Plot — Variance Explained by Each Principal Component")
    plt.tight_layout()
    plt.savefig(f"Task2/charts2/scree_plot.png")
    plt.close()
    return explained, cumulative


def plot_pc_scatter(pc_scores, positions):
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_positions = sorted(set(positions))
    colors = plt.colormaps.get_cmap("tab20").resampled(len(unique_positions))

    for i, pos in enumerate(unique_positions):
        mask = positions == pos
        ax.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                   s=8, alpha=0.6, color=colors(i), label=pos)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("FIFA 19 Players Projected onto the First Two Principal Components")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, ncol=2, title="Position")
    plt.tight_layout()
    plt.savefig(f"Task2/charts2/pc1_pc2_scatter.png", bbox_inches="tight")
    plt.close()

# main function
def main():
    # I. Load & preprocess
    df, X, positions, names = load_and_preprocess()

    # II. Standardize
    Z, mean, std = standardize(X)

    # III. Covariance matrix
    cov = covariance_matrix(Z)
    print("\nCovariance (correlation) matrix:\n", np.round(cov, 2))

    # IV. Eigen-decomposition
    eigenvalues, eigenvectors = eigen_decompose(cov)

    # V. Sort components
    eigenvalues, eigenvectors = sort_components(eigenvalues, eigenvectors)
    print("\nEigenvalues (sorted, descending):\n", np.round(eigenvalues, 3))

    explained, cumulative = plot_scree(eigenvalues)
    print("\nExplained variance %:", np.round(explained, 1))
    print("Cumulative variance %:", np.round(cumulative, 1))

    # VI. Project onto the first 2 components for visualization
    pc_scores = project_data(Z, eigenvectors, n_components=2)
    plot_pc_scatter(pc_scores, positions)

    # Show which original features load most heavily onto PC1 and PC2 —
    # this is what lets us name the components (e.g. "attacking vs defensive")
    print("\nFeature loadings on PC1 and PC2:")
    for i, feat in enumerate(ALL_FEATURES):
        print(f"  {feat:16s}  PC1={eigenvectors[i,0]:+.3f}   PC2={eigenvectors[i,1]:+.3f}")

    print(f"\n[pca] charts saved to /charts2")
    return {
        "df": df, "X": X, "Z": Z, "cov": cov,
        "eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
        "pc_scores": pc_scores, "explained": explained, "cumulative": cumulative,
    }


if __name__ == "__main__":
    main()