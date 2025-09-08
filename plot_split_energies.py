import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import numpy as np
import pickle 


t = 300
d = '0.82'

input_path = f"/Users/arghavan/lab/hp-results/split_energy_results/{t}K/{d}/"
output_path =f"{input_path}/plots/"
mean_nbr_path = f'/Users/arghavan/Graduate Center Dropbox/Arghavan Vedadi Gargari/MyFiles/{t}K/{d}/all_frame_mean_neighbours.pkl'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ------------------------------ Plotting the raw energies  ---------------------------------------
mean_nbr = []
with open(mean_nbr_path, 'rb') as f:
    while True:
        try:
            mean_nbr.append(pickle.load(f))
        except EOFError:
            break
N_nbr = mean_nbr[0][0]

def load_dx_file(dx_file):
    with open(dx_file, "r") as f:
        energies = f.readlines()[4:]
        return np.array([float(value) for e in energies for value in e.split()])
    
bin_width = 0.06

# ---------- left energies ----------

energy_file_left_same_side = f"{input_path}energies-left-same-side.dat"
energy_file_left_opposite_side = f"{input_path}energies-left-opposite-side.dat"

E_left_same_side = load_dx_file(energy_file_left_same_side)
E_left_opposite_side = load_dx_file(energy_file_left_opposite_side)

E_min_left = min(np.min(E_left_same_side), np.min(E_left_opposite_side))
E_max_left = max(np.max(E_left_same_side), np.max(E_left_opposite_side))

left_bins = np.arange(np.floor(E_min_left / bin_width) * bin_width, 
                 np.ceil(E_max_left / bin_width) * bin_width + bin_width, bin_width)

left_same_side_counts, left_edges = np.histogram(E_left_same_side, bins=left_bins)
left_opposite_side_counts, _ = np.histogram(E_left_opposite_side, bins=left_bins)

prob_left_same_side = left_same_side_counts / (bin_width * np.sum(left_same_side_counts))
prob_left_opposite_side = left_opposite_side_counts / (bin_width * np.sum(left_opposite_side_counts))

rho_i_left_same_side = prob_left_same_side * N_nbr
rho_i_left_opposite_side = prob_left_opposite_side * N_nbr

left_bin_centers = left_edges[:-1] + bin_width / 2

# Plot 1: Number density
plt.figure(figsize=(8, 6))
plt.plot(left_bin_centers, rho_i_left_same_side, color='blue', linewidth=2, label="same side energies")
plt.plot(left_bin_centers, rho_i_left_opposite_side, color='green', linewidth=2, label="opposite side energies")
plt.title(f"Number Density Distribution of Pairwise Energies Involving Left Side Waters")
plt.suptitle(f'Plates Distance = {float(d)*10:.1f} Å', fontsize=14)
plt.xlabel(r"$E_n (kcal/mol)$", fontsize=14)
plt.ylabel(r"$\rho_i(E_n)$", fontsize=14)
plt.legend()
plt.xlim(-4.0, 2.5)
plt.ylim(bottom=0)
plt.xticks([-4.0, -2.0, 0.0, 2.0])
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tick_params(axis='x', which='minor', length=2)
plt.tight_layout()
plt.savefig(f"{output_path}/numDens-left-side.png", dpi=300)
plt.close()


# Plot 2: Probability
plt.figure(figsize=(8, 6))
plt.plot(left_bin_centers, prob_left_same_side, color='blue', linewidth=2, label="same side energies")
plt.plot(left_bin_centers, prob_left_opposite_side, color='green', linewidth=2, label="opposite side energies")
plt.title(f"Probability Distribution of Pairwise Energies Involving Left Side Waters")
plt.suptitle(f'Plates Distance = {float(d)*10:.1f} Å', fontsize=14)
plt.xlabel(r"$E_n (kcal/mol)$", fontsize=14)
plt.ylabel(r"$\rho_i(E_n)/N_{{nbr}}$", fontsize=14)
plt.legend()
plt.xlim(-4.0, 2.5)
plt.ylim(bottom=0)
plt.xticks([-4.0, -2.0, 0.0, 2.0])  
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tick_params(axis='x', which='minor', length=2)
plt.tight_layout()
plt.savefig(f"{output_path}/numDens-left-side.png", dpi=300)
plt.close()











# ---------- right energies ----------

energy_file_right_same_side = f"{input_path}energies-right-same-side.dat"
energy_file_right_opposite_side = f"{input_path}energies-right-opposite-side.dat"

E_right_same_side = load_dx_file(energy_file_right_same_side)
E_right_opposite_side = load_dx_file(energy_file_right_opposite_side)

E_min_right = min(np.min(E_right_same_side), np.min(E_right_opposite_side))
E_max_right = max(np.max(E_right_same_side), np.max(E_right_opposite_side))

right_bins = np.arange(np.floor(E_min_right / bin_width) * bin_width,
                 np.ceil(E_max_right / bin_width) * bin_width + bin_width, bin_width)

right_same_side_counts, right_edges = np.histogram(E_right_same_side, bins=right_bins)
right_opposite_side_counts, _ = np.histogram(E_right_opposite_side, bins=right_bins)

prob_right_opposite_side = right_opposite_side_counts / (bin_width * np.sum(right_opposite_side_counts))
prob_right_same_side = right_same_side_counts / (bin_width * np.sum(right_same_side_counts))

rho_i_right_same_side = prob_right_same_side * N_nbr
rho_i_right_opposite_side = prob_right_opposite_side * N_nbr

right_bin_centers = right_edges[:-1] + bin_width / 2



# ----- Plotting from the prob. and numdens files  ----------


plt.figure(figsize=(8, 6))
plt.plot(x_axis, no_wats, color='blue', linewidth=2, label=f"Plates distance = {float_d*10:.1f} Å")
# plt.plot(bin_centers, rho_i_plate, color='green', linewidth=2, label=f"{float(d)*10:.1f} Å")
plt.title("Water Distribution between the edges")
plt.xlabel('Distance from the edge in Å', fontsize=14)
plt.ylabel('Average Count', fontsize=14)
plt.legend()
plt.xlim(0, 18)
plt.ylim(bottom=0)
# plt.xticks([-4.0, -2.0, 0.0, 2.0])
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tick_params(axis='x', which='minor', length=2)
plt.tight_layout()
plt.savefig(f"{output_path}/counts-y.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 6))
plt.plot(x_axis, numdens, color='blue', linewidth=2, label=f"Plates distance = {float_d*10:.1f} Å")
plt.title("Number Density Distribution between the edges")
plt.xlabel('Distance from the edge in Å', fontsize=14)
plt.ylabel('Number Density N/(Å^3)', fontsize=14)
plt.legend()
plt.xlim(0, 18)
plt.ylim(bottom=0)
# plt.xticks([-4.0, -2.0, 0.0, 2.0])
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.tick_params(axis='x', which='minor', length=2)
plt.tight_layout()
plt.savefig(f"{output_path}/numDens-y.png", dpi=300)
plt.close()