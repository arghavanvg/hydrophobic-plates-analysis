"""

_summary_

"""

import mdtraj as md
import numpy as np
from scipy.spatial.distance import pdist, squareform
import parmed as pmd
from warnings import filterwarnings
filterwarnings('ignore')
import logging
import os
import sys
import pickle


# -------------------- Loading the trajectory and topology files --------------------
t = int(sys.argv[1])
d = sys.argv[2]

input_path = f'/Users/arghavan/Graduate Center Dropbox/Arghavan Vedadi Gargari/MyFiles/{t}K/{d}/'
# hbond_output_path = f'/Users/arghavan/lab/hp-results/hbond_results/{t}K/{d}/'
# nbr_output_path = f'/Users/arghavan/lab/hp-results/nbr_results/{t}K/{d}/'
energy_output_path = f'/Users/arghavan/lab/hp-results/energy_results/{t}K/{d}/'


# prob_file = f'{energy_output_path}probability_{t}K_{d}.dat'
# numDens_file = f'{energy_output_path}numDens_{t}K_{d}.dat'
# energy_dx_file = f'{energy_output_path}pairwise_E_{t}K_{d}.dx'
logfile = f'{energy_output_path}log_{t}K_{d}.log'

# nbr_file = f'{nbr_output_path}nbrs_{t}K_{d}.dat'
# hbond_file = f'{hbond_output_path}hbonds_{t}K_{d}.dat'
mean_nbr_path = f'{input_path}all_frame_mean_neighbours.pkl'


traj_path = f'{input_path}{t}K_{d}.nc'
top_path = f'{input_path}topol_edited.prmtop'

traj = md.load(traj_path, top=top_path)
tpl = traj.top
parm = pmd.load_file(top_path)

all_ox_tpl_indices = tpl.select("name O")
all_wat_atm_tpl_indices = tpl.select("water")
walls = tpl.select("resname =~ 'WALL'")

n_frames = traj.n_frames
# n_frames = 50

# --------------------------- Logging Setup ---------------------------

os.makedirs(os.path.dirname(logfile), exist_ok=True)

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

file_handler = FlushFileHandler(logfile, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%I:%M:%S %p'))
logger.addHandler(file_handler)


# --------------------------- Constants ---------------------------

Acoeff = np.array(parm.parm_data['LENNARD_JONES_ACOEF'][0], dtype=float)
Bcoeff = np.array(parm.parm_data['LENNARD_JONES_BCOEF'][0], dtype=float)
chg_lst = np.array(parm.parm_data['CHARGE'][:4], dtype=float)*18.2223

o_charge = chg_lst[3]  
h_charge = chg_lst[1]

oo_chg = o_charge**2
hh_chg = h_charge**2
oh_chg = o_charge * h_charge


# --------------------------- Surface Between the Plates ---------------------------
wall_coords = traj.xyz[0][walls]

float_d = float(d)
edge = 0.230  # A
x_center = 2.50
x_min, x_max = np.float32(x_center - float_d/2), np.float32(x_center + float_d/2)
y_min, y_max = np.min(wall_coords[:,1]) + edge , np.max(wall_coords[:,1]) - edge
z_min, z_max = np.min(wall_coords[:,2]) + edge , np.max(wall_coords[:,2]) - edge


radius = 3.5 #A

nbrs = []
# pairwise_energies = []
num_wat_in_box = []

E_left_left = []
E_right_right = []
E_left_right = []
E_right_left = []

for frame_num in range(n_frames):
     
    logger.info(f"Processing frame {frame_num + 1}")

    # ------------------------------ Masking ---------------------------------------


    all_ox_coords = traj.xyz[frame_num][all_ox_tpl_indices]
    mask = (
    (all_ox_coords[:, 0] >= x_min) & (all_ox_coords[:, 0] <= x_max) &
    (all_ox_coords[:, 1] >= y_min) & (all_ox_coords[:, 1] <= y_max) &
    (all_ox_coords[:, 2] >= z_min) & (all_ox_coords[:, 2] <= z_max)
)


    selected_ox_tpl_indices = all_ox_tpl_indices[mask]
    selected_wat_tpl_indices = [x for item in selected_ox_tpl_indices for x in range(item, item + 4)]
    index_map = {value: idx for idx, value in enumerate(selected_wat_tpl_indices)}
    num_wat_in_box.append(len(selected_ox_tpl_indices))

    left_ox_indices = selected_ox_tpl_indices[all_ox_coords[mask][:, 0] < x_center]
    right_ox_indices = selected_ox_tpl_indices[all_ox_coords[mask][:, 0] >= x_center]

    # Store energies
    left_left_energies = [] 
    right_right_energies = []
    left_right_energies = []
    right_left_energies = []

    # ------------------------------ Distance Matrix ---------------------------------------
    coords = traj.xyz[frame_num][selected_wat_tpl_indices]

    dist_matrix_in_nm = squareform(pdist(coords, metric='euclidean'))
    dist_matrix = dist_matrix_in_nm * 10

    # ------------------------------ Pairwise Energy Calculation ---------------------------------------
    frame_energies = [] 
    computed_pairs = set()
    frame_nbrs = []

    for i in selected_ox_tpl_indices:
        for j in selected_ox_tpl_indices:

            pair = frozenset((i, j))

            if pair in computed_pairs or i >= j:
                continue
            
            d_i = index_map[i]
            d_j = index_map[j]
            
            if dist_matrix[d_i][d_j] > radius:
                continue

            lj_energy = (Acoeff / dist_matrix[d_i][d_j]** 12) - (Bcoeff / dist_matrix[d_i][d_j]** 6)
            electrostatic_energy = (
            oo_chg / dist_matrix[d_i+3][d_j+3] + oh_chg / dist_matrix[d_i+3][d_j+1] + oh_chg / dist_matrix[d_i+3][d_j+2] +
            oh_chg / dist_matrix[d_i+1][d_j+3] + hh_chg / dist_matrix[d_i+1][d_j+1] + hh_chg / dist_matrix[d_i+1][d_j+2] +
            oh_chg / dist_matrix[d_i+2][d_j+3] + hh_chg / dist_matrix[d_i+2][d_j+1] + hh_chg / dist_matrix[d_i+2][d_j+2]
            )
            total_energy = (lj_energy + electrostatic_energy) / 2

            if i in left_ox_indices and j in left_ox_indices:
                left_left_energies.append(total_energy)
            elif i in right_ox_indices and j in right_ox_indices:
                right_right_energies.append(total_energy)
            elif i in left_ox_indices and j in right_ox_indices:
                left_right_energies.append(total_energy)
            elif i in right_ox_indices and j in left_ox_indices:
                right_left_energies.append(total_energy)
            computed_pairs.add(pair)
    # pairwise_energies.extend(frame_energies)
    E_left_left.extend(left_left_energies)
    E_right_right.extend(right_right_energies)
    E_left_right.extend(left_right_energies)
    E_right_left.extend(right_left_energies)

# pairwise_energies = np.array(pairwise_energies)
avg_num_wat_in_box = np.mean(num_wat_in_box)


header = f"#[laptop input] Pairwise energy: {n_frames} frames for {t}K - {float(d)*10} A simulation \n# Column: E_n (kcal/mol)"
# np.savetxt(energy_dx_file, pairwise_energies, fmt="%.6f", header=header,  delimiter=" ")
np.savetxt(f"{energy_output_path}left_left_energy.dat", np.array(E_left_left), fmt="%.6f", header="Left-Left Energies\n" + header)
np.savetxt(f"{energy_output_path}right_right_energy.dat", np.array(E_right_right), fmt="%.6f", header="Right-Right Energies\n" + header)
np.savetxt(f"{energy_output_path}left_right_energy.dat", np.array(E_left_right), fmt="%.6f", header="Left-Right Energies\n" + header)
np.savetxt(f"{energy_output_path}right_left_energy.dat", np.array(E_right_left), fmt="%.6f", header="Right-Left Energies\n" + header)


# logger.info(f"Saved in {energy_dx_file}")
logger.info(f"{n_frames} frames - split_space_in_two.py - plates E + Plot ")
# logger.info(f"{len(pairwise_energies)} pairwise energies calculated.")
logger.info(f"{avg_num_wat_in_box} average No. of waters in the box.")



# --------------------------- Data for plotting ---------------------------
mean_nbr = []
with open(mean_nbr_path, 'rb') as f:
    while True:
        try:
            mean_nbr.append(pickle.load(f))
        except EOFError:
            break
N_nbr = mean_nbr[0][0]


bin_width = 0.06  # kcal/mol

global_min_E = min(np.min(E_left_left), np.min(E_right_right), np.min(E_left_right), np.min(E_right_left))
global_max_E = max(np.max(E_left_left), np.max(E_right_right), np.max(E_left_right), np.max(E_right_left))


bins = np.arange(np.floor(global_min_E / bin_width) * bin_width, 
                 np.ceil(global_max_E / bin_width) * bin_width + bin_width, bin_width)

left_left_counts, edges = np.histogram(E_left_left, bins=bins)
right_right_counts, edges = np.histogram(E_right_right, bins=bins)
left_right_counts, edges = np.histogram(E_left_right, bins=bins)
right_left_counts, edges = np.histogram(E_right_left, bins=bins)

bin_centers = edges[:-1] + bin_width / 2

prob_left_left = left_left_counts / (bin_width * np.sum(left_left_counts))
rho_i_left_left = prob_left_left * N_nbr
prob_right_right = right_right_counts / (bin_width * np.sum(right_right_counts))
rho_i_right_right = prob_right_right * N_nbr
prob_left_right = left_right_counts / (bin_width * np.sum(left_right_counts))
rho_i_left_right = prob_left_right * N_nbr
prob_right_left = right_left_counts / (bin_width * np.sum(right_left_counts))
rho_i_right_left = prob_right_left * N_nbr




with open(f'{energy_output_path}probability_left_left.dat', "w") as f:
    for x, y in zip(bin_centers, prob_left_left):
        f.write(f"{x:.4f} {y:.6f}\n")
with open(f'{energy_output_path}numDens_left_left.dat', "w") as f:
    for x, y in zip(bin_centers, rho_i_left_left):
        f.write(f"{x:.4f} {y:.6f}\n")

with open(f'{energy_output_path}probability_right_right.dat', "w") as f:
    for x, y in zip(bin_centers, prob_right_right):
        f.write(f"{x:.4f} {y:.6f}\n")
with open(f'{energy_output_path}numDens_right_right.dat', "w") as f:
    for x, y in zip(bin_centers, rho_i_right_right):
        f.write(f"{x:.4f} {y:.6f}\n")

with open(f'{energy_output_path}probability_left_right.dat', "w") as f:
    for x, y in zip(bin_centers, prob_left_right):
        f.write(f"{x:.4f} {y:.6f}\n")
with open(f'{energy_output_path}numDens_left_right.dat', "w") as f:
    for x, y in zip(bin_centers, rho_i_left_right):
        f.write(f"{x:.4f} {y:.6f}\n")

with open(f'{energy_output_path}probability_right_left.dat', "w") as f:
    for x, y in zip(bin_centers, prob_right_left):
        f.write(f"{x:.4f} {y:.6f}\n")
with open(f'{energy_output_path}numDens_right_left.dat', "w") as f:
    for x, y in zip(bin_centers, rho_i_right_left):
        f.write(f"{x:.4f} {y:.6f}\n")