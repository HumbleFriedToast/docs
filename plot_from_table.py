"""
Trace les temps mesurés (provenant du tableau fourni) et les compare aux courbes théoriques.
Génère des PNG pour chaque algorithme ainsi qu'un graphique combiné.
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Données fournies (N et temps mesurés T_A1..T_A4)
N = np.array([
    1000003,
    2000003,
    4000037,
    8000009,
    16000057,
    32000011,
    64000031,
    128000003,
    256000001,
    512000009,
    1024000009,
    2048000011,
], dtype=float)

T_A1 = np.array([
    0.002288,
    0.002768,
    0.005613,
    0.011118,
    0.022654,
    0.044510,
    0.088886,
    0.177057,
    0.353173,
    0.716674,
    1.432980,
    2.870783,
], dtype=float)

T_A2 = np.array([
    0.000695,
    0.001396,
    0.002825,
    0.005549,
    0.011797,
    0.022150,
    0.043949,
    0.088214,
    0.176134,
    0.359405,
    0.724569,
    1.440214,
], dtype=float)

T_A3 = np.array([
    0.000004,
    0.000002,
    0.000003,
    0.000005,
    0.000006,
    0.000008,
    0.000012,
    0.000016,
    0.000022,
    0.000033,
    0.000047,
    0.000065,
], dtype=float)

T_A4 = np.array([
    0.000001,
    0.000001,
    0.000001,
    0.000002,
    0.000002,
    0.000003,
    0.000004,
    0.000006,
    0.000008,
    0.000011,
    0.000016,
    0.000022,
], dtype=float)

# (Nom affiché, temps mesuré, fonction théorique, couleur)
algorithms = [
    ("A1 - naïf (O(n))", T_A1, lambda x: x, 'tab:blue'),
    ("A2 - n/2 (O(n))", T_A2, lambda x: x/2.0, 'tab:orange'),
    ("A3 - parité (O(n) avec constante réduite)", T_A3, lambda x: x, 'tab:green'),
    ("A4 - racine (O(√n))", T_A4, np.sqrt, 'tab:red'),
]

out_dir = os.path.dirname(__file__) or '.'

# Graphique individuel pour chaque algorithme
for name, measured, theory_fn, color in algorithms:
    plt.figure(figsize=(8, 5))
    plt.loglog(N, measured, marker='o', linestyle='-', color=color, label='mesuré')

    # Calcul de la courbe théorique, ajustée pour l’échelle
    th = theory_fn(N.astype(float))
    last_idx = -1
    scale = measured[last_idx] / float(th[last_idx]) if th[last_idx] != 0 else 1.0
    th_scaled = th * scale
    plt.loglog(N, th_scaled, linestyle='--', color='gray', label='théorique (échelle ajustée)')

    plt.xlabel('Taille de l’entrée N (échelle logarithmique)')
    plt.ylabel('Temps (secondes, échelle logarithmique)')
    plt.title(f'Comparaison temps mesuré vs théorie — {name}')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()

    # Ajustement automatique de l’axe Y
    measured_mask = ~np.isnan(measured)
    if measured_mask.any():
        mmin = measured[measured_mask].min()
        mmax = measured[measured_mask].max()
        ylow = max(mmin * 0.3, mmin / 10.0)
        yhigh = mmax * 3.0
        if ylow > 0 and yhigh > ylow:
            plt.ylim(ylow, yhigh)

    fname = os.path.join(out_dir, f"table_plot_{name.split()[0]}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Image enregistrée : {fname}")

# Graphique combiné (tous les algorithmes)
plt.figure(figsize=(10, 6))
plt.loglog(N, T_A1, marker='o', label='A1 mesuré')
plt.loglog(N, T_A2, marker='o', label='A2 mesuré')
plt.loglog(N, T_A3, marker='o', label='A3 mesuré')
plt.loglog(N, T_A4, marker='o', label='A4 mesuré')

# Ajouter courbes théoriques avec mise à l'échelle
for name, measured, theory_fn, color in algorithms:
    th = theory_fn(N.astype(float))
    last_idx = -1
    scale = measured[last_idx] / (float(th[last_idx]) if th[last_idx] != 0 else 1.0)
    plt.loglog(N, th * scale, linestyle='--', color='gray', alpha=0.6, label=f'{name} théorie (échelle)')

plt.xlabel('Taille de l’entrée N (échelle logarithmique)')
plt.ylabel('Temps (secondes, échelle logarithmique)')
plt.title('Comparaison globale des algorithmes (log-log)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
combined_out = os.path.join(out_dir, 'table_combined.png')
plt.savefig(combined_out)
plt.close()
print(f"Image enregistrée : {combined_out}")
