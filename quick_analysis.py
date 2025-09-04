import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_excel('soybean_master_data.xlsx')
df = df[['Variety', 'Average R', 'Average G', 'Average B']].dropna()

print("=== SOYBEAN DATA ANALYSIS ===")
print(f"Total samples: {len(df)}")
print("\nVariety distribution:")
print(df['Variety'].value_counts())

print("\nAverage RGB values by variety:")
avg_rgb = df.groupby('Variety')[['Average R', 'Average G', 'Average B']].mean()
print(avg_rgb)

print("\nRGB value ranges by variety:")
for variety in df['Variety'].unique():
    variety_data = df[df['Variety'] == variety]
    print(f"\nVariety {variety}:")
    print(f"  R: {variety_data['Average R'].min():.1f} - {variety_data['Average R'].max():.1f}")
    print(f"  G: {variety_data['Average G'].min():.1f} - {variety_data['Average G'].max():.1f}")
    print(f"  B: {variety_data['Average B'].min():.1f} - {variety_data['Average B'].max():.1f}")

# Create detailed plots
plt.figure(figsize=(15, 10))

# 1. RGB values by variety
plt.subplot(2, 2, 1)
colors = {'1110': 'red', '1135': 'green', '2172': 'blue'}
for variety, color in colors.items():
    variety_data = df[df['Variety'] == int(variety)]
    plt.scatter(variety_data['Average R'], variety_data['Average G'], 
               alpha=0.7, label=variety, color=color, s=50)
plt.xlabel('Average R')
plt.ylabel('Average G')
plt.title('R vs G - Variety Separation')
plt.legend()
plt.grid(True)

# 2. Brightness distribution
plt.subplot(2, 2, 2)
df['Brightness'] = (df['Average R'] + df['Average G'] + df['Average B']) / 3
for variety, color in colors.items():
    variety_data = df[df['Variety'] == int(variety)]
    plt.hist(variety_data['Brightness'], alpha=0.6, label=variety, 
             color=color, bins=20, density=True)
plt.xlabel('Brightness')
plt.ylabel('Density')
plt.title('Brightness Distribution')
plt.legend()

# 3. R/G Ratio
plt.subplot(2, 2, 3)
for variety, color in colors.items():
    variety_data = df[df['Variety'] == int(variety)]
    r_g_ratio = variety_data['Average R'] / variety_data['Average G']
    plt.hist(r_g_ratio, alpha=0.6, label=variety, color=color, bins=15, density=True)
plt.xlabel('R/G Ratio')
plt.ylabel('Density')
plt.title('R/G Ratio Distribution')
plt.legend()

# 4. 3D plot (if you have matplotlib 3D)
try:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(2, 2, 4, projection='3d')
    for variety, color in colors.items():
        variety_data = df[df['Variety'] == int(variety)]
        ax.scatter(variety_data['Average R'], variety_data['Average G'], variety_data['Average B'], 
                  alpha=0.6, label=variety, color=color, s=30)
    ax.set_xlabel('Average R')
    ax.set_ylabel('Average G')
    ax.set_zlabel('Average B')
    ax.set_title('3D RGB Space')
    ax.legend()
except:
    # Alternative 2D plot
    plt.subplot(2, 2, 4)
    for variety, color in colors.items():
        variety_data = df[df['Variety'] == int(variety)]
        plt.scatter(variety_data['Average R'], variety_data['Average B'], 
                   alpha=0.6, label=variety, color=color, s=50)
    plt.xlabel('Average R')
    plt.ylabel('Average B')
    plt.title('R vs B - Variety Separation')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('soybean_variety_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalysis complete! Check 'soybean_variety_analysis.png' for visualizations.")