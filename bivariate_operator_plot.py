import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import factorial
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

def f(x, y):
    """Test function: f(x, y) = x^2 + y^2 + xy"""
    return x**2 + y**2 + x*y

def K(y1, y2, m=2, n=2, eta1=1.0, eta2=1.0, alpha1=1.0, alpha2=1.0, Kmax=20):
    """
    Bivariate operator K_{m,n,eta1,eta2}^{alpha1,alpha2}(f; y1, y2)
    """
    total = 0.0
    for k1 in range(Kmax + 1):
        for k2 in range(Kmax + 1):
            # Poisson weight
            weight = np.exp(-(m*y1 + n*y2)) * ((m*y1)**k1 / factorial(k1)) * ((n*y2)**k2 / factorial(k2))
            
            # Double integral
            def integrand1(s1):
                def integrand2(s2):
                    x = (k1 + s1**eta1) / (m + alpha1)
                    y = (k2 + s2**eta2) / (n + alpha2)
                    return f(x, y)
                return quad(integrand2, 0, 1)[0]
            
            total += weight * quad(integrand1, 0, 1)[0]
    
    return total

# Parameters
params = [
    {'m': 100, 'n': 100, 'name': 'm=n=100'},
    {'m': 200, 'n': 200, 'name': 'm=n=200'},
    {'m': 300, 'n': 300, 'name': 'm=n=300'}
]

alpha1 = alpha2 = 1.0
eta1 = eta2 = 1.0

# Plot for each parameter set
for param_set in params:
    m = param_set['m']
    n = param_set['n']
    name = param_set['name']
    
    print(f"\nComputing operator for {name}...")
    
    # Grid
    y1_vals = np.linspace(0, 1, 50)
    y2_vals = np.linspace(0, 1, 50)
    Y1, Y2 = np.meshgrid(y1_vals, y2_vals)
    
    # Compute Z values
    Z = np.zeros_like(Y1)
    for i in range(len(y1_vals)):
        for j in range(len(y2_vals)):
            Z[j, i] = K(y1_vals[i], y2_vals[j], m=m, n=n, eta1=eta1, eta2=eta2, alpha1=alpha1, alpha2=alpha2, Kmax=20)
            if (i + 1) * (j + 1) % 250 == 0:
                print(f"  Progress: {((i + 1) * (j + 1) / (len(y1_vals) * len(y2_vals)) * 100):.1f}%")
    
    print(f"  Done! Z range: [{Z.min():.4f}, {Z.max():.4f}]")
    
    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(Y1, Y2, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('y₁')
    ax1.set_ylabel('y₂')
    ax1.set_zlabel('K(f; y₁, y₂)')
    ax1.set_title(f'3D Surface Plot: {name}\nα₁=α₂=1, η₁=η₂=1')
    fig.colorbar(surf, ax=ax1, label='K value')
    
    # Contour Plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(Y1, Y2, Z, levels=30, cmap='viridis')
    ax2.contour(Y1, Y2, Z, levels=15, colors='black', alpha=0.2, linewidths=0.5)
    ax2.set_xlabel('y₁')
    ax2.set_ylabel('y₂')
    ax2.set_title(f'Contour Plot: {name}\nα₁=α₂=1, η₁=η₂=1')
    fig.colorbar(contour, ax=ax2, label='K value')
    
    plt.tight_layout()
    filename = f'bivariate_operator_{name.replace("=", "").replace(",", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.show()

print("\n✓ All plots generated successfully!")
