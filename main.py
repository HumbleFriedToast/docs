"""
Mesure et comparaison de plusieurs algorithmes simples de test de primalité.

Algorithmes inclus :
 - is_prime_naive : test de divisibilité de 2 à n-1 (méthode naïve)
 - is_prime_n_over_2 : test de 2 à n//2
 - is_prime_parity : élimine la parité, puis teste les diviseurs impairs
 - is_prime_sqrt : teste les diviseurs jusqu'à sqrt(n)

Le script mesure ces implémentations sur des entrées pires-cas (n premiers)
et produit des graphiques comparant les temps mesurés aux courbes
théoriques mise à l'échelle.
"""

import math
import time
import statistics
import os
from typing import Callable, List, Optional

import matplotlib
# use a non-interactive backend so the script can run headless and save files
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def is_prime_naive(n: int) -> bool:
	"""Naive check: test divisibility by every integer from 2 to n-1.

	Time complexity: O(n)
	"""
	if n <= 1:
		return False
	if n <= 3:
		return True
	for d in range(2, n):
		if n % d == 0:
			return False
	return True


def is_prime_n_over_2(n: int) -> bool:
	"""Check divisibility by integers from 2 to n//2.

	Time complexity: O(n) but ~half the checks of the full naive method.
	"""
	if n <= 1:
		return False
	if n <= 3:
		return True
	limit = n // 2
	for d in range(2, limit + 1):
		if n % d == 0:
			return False
	return True


def is_prime_parity(n: int) -> bool:
	"""Parity-optimized: eliminate even numbers first, then check odd divisors
	up to n-1 (so still O(n) but checks only odd divisors).
	"""
	if n <= 1:
		return False
	if n == 2:
		return True
	if n % 2 == 0:
		return False
	# check odd divisors
	for d in range(3, n, 2):
		if n % d == 0:
			return False
	return True


def is_prime_sqrt(n: int) -> bool:
	"""Check divisibility up to sqrt(n). Typical efficient simple test.

	Time complexity: O(sqrt(n))
	"""
	if n <= 1:
		return False
	if n <= 3:
		return True
	if n % 2 == 0:
		return n == 2
	limit = int(math.isqrt(n))
	for d in range(3, limit + 1, 2):
		if n % d == 0:
			return False
	return True


def next_prime_at_least(n: int) -> int:
	"""Return the smallest prime >= n using the sqrt test (fast enough for our needs)."""
	if n <= 2:
		return 2
	candidate = n if n % 2 == 1 else n + 1
	while True:
		if is_prime_sqrt(candidate):
			return candidate
		candidate += 2


def measure_time(func: Callable[[int], bool], n: int, repeats: int = 3, timeout: float = 10.0) -> Optional[float]:
	"""Measure average execution time of func(n) (in seconds).

	If a single run exceeds `timeout` seconds, returns None to indicate skip.
	"""
	times: List[float] = []
	for _ in range(repeats):
		t0 = time.perf_counter()
		func(n)
		t1 = time.perf_counter()
		elapsed = t1 - t0
		times.append(elapsed)
		if elapsed > timeout:
			return None
	# return median to reduce noise
	return statistics.median(times)


def run_benchmarks():
	# use the exact list requested by the user (may be large). We'll ensure
	# inputs are prime by taking the next prime >= each listed value.
	targets = [
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
	]
	print("Préparation des entrées premières (pire cas pour les boucles)...")
	primes = [next_prime_at_least(t) for t in targets]
	# vérification : s'assurer que toutes les valeurs choisies sont bien premières
	for p in primes:
		if not is_prime_sqrt(p):
			raise RuntimeError(f"Entrée sélectionnée {p} n'est pas première — interruption.")
	print("Entrées (n premières) :", primes)

	algorithms = [
		("naïf (2..n-1)", is_prime_naive),
		("n/2 (2..n/2)", is_prime_n_over_2),
		("parité (impairs)", is_prime_parity),
		("racine (<= sqrt(n))", is_prime_sqrt),
	]

	results = {name: [] for name, _ in algorithms}

	# for these large sizes, use fewer repeats and a reasonable timeout so
	# extremely slow algorithms are skipped instead of running indefinitely.
	repeats = 1
	timeout = 5.0

	for n in primes:
		print(f"\nMesure pour n = {n}")
		for name, func in algorithms:
			print(f" - Exécution de {name}...", end=" ")
			t = measure_time(func, n, repeats=repeats, timeout=timeout)
			if t is None:
				print("ignoré (trop lent)")
				results[name].append(float('nan'))
			else:
				print(f"{t:.6f} s")
				results[name].append(t)

	# Prepare per-algorithm plots
	folder = os.path.dirname(__file__) or "."
	n_vals = np.array(primes)

	# precompute theoretical shapes
	th_n = n_vals.astype(float)
	th_n_over_2 = n_vals.astype(float) / 2.0
	th_sqrt = np.sqrt(n_vals.astype(float))

	largest_idx = -1
	for name, func in algorithms:
		times = np.array(results[name], dtype=float)
		mask = ~np.isnan(times)

		safe_name = name.replace(' ', '_').replace('/', '_').replace('..', '-')
		out_file = os.path.join(folder, f"comparison_{safe_name}.png")

		plt.figure(figsize=(8, 5))
		if mask.any():
			plt.plot(n_vals[mask], times[mask], marker='o', linestyle='-', label='mesuré')
		else:
			# pas de données mesurées disponibles (toutes ignorées); afficher la théorie
			print(f"Pas de données mesurées pour {name} ; tracé théorique seulement.")

		# choisir la courbe théorique correspondant à l'algorithme
		if 'racine' in name or 'sqrt' in name:
			fvals = th_sqrt
			theory_label = 'O(sqrt(n))'
		elif 'n/2' in name:
			fvals = th_n_over_2
			theory_label = 'O(n/2)'
		else:
			# naïf et parité sont linéaires en n (parité réduit le facteur constant)
			fvals = th_n
			theory_label = 'O(n)'

		# mise à l'échelle de la courbe théorique pour correspondre à la dernière valeur mesurée
		scale = 1.0
		if mask.any():
			last_idx = np.where(mask)[0][-1]
			denom = float(fvals[last_idx]) if fvals[last_idx] != 0 else 1.0
			scale = float(times[last_idx]) / denom
		else:
			# échelle arbitraire si aucune mesure
			scale = 1e-6

		plt.plot(n_vals, fvals * scale, linestyle='--', color='C1', label=f'théorique (mise à l\'échelle) : {theory_label}')

		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel('n (échelle logarithmique)')
		plt.ylabel("temps (s, échelle logarithmique)")
		plt.title(f'Mesuré vs Théorique — {name}')
		plt.legend()
		plt.grid(True, which='both', ls='--', alpha=0.4)

		# zoomer l'axe y autour des valeurs mesurées pour rendre les différences plus visibles
		if mask.any():
			measured_min = float(np.min(times[mask]))
			measured_max = float(np.max(times[mask]))
			# ajouter un peu de marge multiplicative pour que la vue sur log reste pertinente
			y_low = max(measured_min * 0.5, measured_min / 10.0)
			y_high = measured_max * 2.0
			if y_low <= 0:
				y_low = measured_min * 0.1 if measured_min > 0 else 1e-12
			plt.ylim(y_low, y_high)

		plt.tight_layout()
		plt.savefig(out_file)
		plt.close()
		print(f"Enregistré le graphique de comparaison pour '{name}' dans {out_file}")

	# Also keep the combined plot for convenience (measured lines for all algorithms)
	combined_out = os.path.join(folder, "big_o_vs_measured.png")
	plt.figure(figsize=(10, 6))
	for name in results:
		times = np.array(results[name], dtype=float)
		mask = ~np.isnan(times)
		if mask.any():
			plt.plot(n_vals[mask], times[mask], marker='o', label=f"mesuré : {name}")

	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('n (échelle logarithmique)')
	plt.ylabel('temps (s, échelle logarithmique)')
	plt.title('Temps mesurés (tous les algorithmes)')
	plt.legend()
	plt.grid(True, which='both', ls='--', alpha=0.4)
	plt.tight_layout()
	plt.savefig(combined_out)
	plt.close()
	print(f"Enregistré le graphique combiné dans {combined_out}")


if __name__ == "__main__":
	run_benchmarks()
