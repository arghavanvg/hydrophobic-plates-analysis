import mdtraj as md
import numpy as np
import os
from decimal import Decimal
from warnings import filterwarnings

filterwarnings("ignore")

T_values = [f"{T}" for T in range(300, 301, 20)]
d_values = [f"{Decimal('0.24') + Decimal('0.02') * i:.2f}" for i in range(16, 64)]

def load_trajectory(input_path: str, temp: str, dist: str):
    traj_path = f"{input_path}{temp}K_{dist}.nc"
    top_path = f"{input_path}topol_edited.prmtop"
    return md.load(traj_path, top=top_path)

def get_plate_boundaries(traj, plate_atom_indices, plate_distance):
    wall_coords = traj.xyz[0][plate_atom_indices]
    x_center = 2.50
    x_min = np.float32(x_center - plate_distance / 2)
    x_max = np.float32(x_center + plate_distance / 2)
    y_min = np.min(wall_coords[:, 1]) 
    y_max = np.max(wall_coords[:, 1]) 
    z_min = np.min(wall_coords[:, 2]) 
    z_max = np.max(wall_coords[:, 2]) 
    return x_min, x_max, y_min, y_max, z_min, z_max

def count_waters_per_frame(traj, ox_indices, bounds):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    num_waters_per_frame = []
    for frame_num in range(traj.n_frames):
        coords = traj.xyz[frame_num][ox_indices]
        in_box = (
            (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
            (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
        )
        num_waters_per_frame.append(np.count_nonzero(in_box))
    return num_waters_per_frame

def main():
    base_input = "/gibbs/arghavan/plate_simulations/"
    base_output = f"/gibbs/arghavan/hp-results-pc/number_of_waters/"
    os.makedirs(base_output, exist_ok=True)
    
    for temp in T_values:
        avg_wat_for_one_t = []
        for dist in d_values:
            try:
                input_path = f"{base_input}{temp}K/{dist}/"
                traj = load_trajectory(input_path, temp, dist)
                tpl = traj.topology
                all_ox_indices = tpl.select("name O")
                plate_indices = tpl.select("resname =~ 'WALL'")

                bounds = get_plate_boundaries(traj, plate_indices, float(dist))
                no_of_waters_between_plates = count_waters_per_frame(traj, all_ox_indices, bounds)
                avg_waters = np.mean(no_of_waters_between_plates)

                avg_wat_for_one_t.append((dist, avg_waters))
                # print(f"✔ {temp}K {dist} nm → Avg waters: {avg_waters:.2f}")
            except Exception as e:
                print(f"✘ Failed for {temp}K {dist} nm: {e}")

        # Save one DX file per temperature
        out_file = os.path.join(base_output, f"{temp}K_tot_num_waters.dx")

        with open(out_file, "w") as f:
            f.write(f"Number of waters in the confined volume, excluding the meniscus and plate volume\n")
            f.write("# Distance(Å)      #N\n")
            for dist, avg in avg_wat_for_one_t:
                f.write(f"{float(dist)*10:.1f}      {avg:.3f}\n")

if __name__ == "__main__":
    main()
