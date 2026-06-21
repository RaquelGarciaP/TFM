from Halo import Halo
from Snapshot import Snapshot

# Files
AHF_file = "./project/data/PLANCK.CLUES.4096.00522.z0.000.AHF_particles"
snapshot_file = "./project/data/PLANCK.CLUES.4096.00522.0"

with open(AHF_file, "r") as f:
    # Read the total number of halos (first line of the file)
    n_halos_total = int(f.readline().strip())
    print(f"Total number of halos on the AHF file: {n_halos_total}")

# Open snapshot file via a Python class
s = Snapshot(snapshot_file)

# If we want to save the properties of all the halos on the snapshot we use the following loop
"""for i in range(n_halos_total):
    halo = Halo(s, AHF_file, i)

    # If we want to save a specific property, we can select as many as we want of the following
    halo.save_contamination_fraction()
    halo.save_cm()
    halo.save_Mg()
    halo.save_Mg_cold_sfr()
    halo.save_Mg_cold_temperature()
    halo.save_Mdm()
    halo.save_Mstars()
    halo.save_Mbh()
    halo.save_Mhalo()
    halo.save_sfr()
    halo.save_sfh()
    halo.save_r50()
    halo.save_r1()
    halo.save_k_co()
    halo.save_vmax()
    halo.save_r_halo()
"""

# If we want to save all the properties above, we can use the following function
# halo.save_all_properties()


# If we want to select only one halo of the snapshot, e.g. halo  with ID = to 3:
halo3 = Halo(s, AHF_file, 3, output_dir="./project/output/1st_simulation/")

# We can also do plots of the halo:
"""halo3.plot_halo()
halo3.plot_stellar_density()
halo3.plot_gas_density()
halo3.plot_gas_temperature()"""
