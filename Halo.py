from functools import cached_property
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
from scipy.optimize import root_scalar

# from Snapshot import Snapshot


class Halo:

    def __init__(self, f, f_AHF, idx, output_dir="./project/output/"):
        # def __init__(Halo: self, Snapshot: f, str: f_AHF, int: idx, str: output_dir="./project/output/"):
        self.snap = f
        self.f = self.snap.f  # h5py.File(f, "r")
        self.n_parts, self.ids_parts = self.__read_AHF(f_AHF, idx)
        self.idx = idx  # halo index
        self.output_dir = output_dir

        # --- Save the INDEXES of the particles inside the halo ---
        # (we save the positions of the particles in the file, not the IDs)
        # Also, the indexes are sorted in ascending order (required by h5py)
        self.idx_gas = self._ids_in_halo(self._load_ids(0), self.ids_parts)
        self.idx_dm = self._ids_in_halo(self._load_ids(1), self.ids_parts)
        self.idx_stars = self._ids_in_halo(self._load_ids(4), self.ids_parts)
        self.idx_bh = self._ids_in_halo(self._load_ids(5), self.ids_parts)

        # Mapping to select the desired particle type when necessary
        self.mapping = {
            0: self.idx_gas,
            1: self.idx_dm,
            4: self.idx_stars,
            5: self.idx_bh,
        }

    # ------- SIMULATION PROPERTIES -------
    @cached_property
    def min_dm_mass(self):
        """Minimum dark matter mass of the simulation."""
        return self.snap.min_dm

    @cached_property
    def min_t(self):
        """Minimum time in which a star is formed in the simulation in [years]."""
        return self.snap.min_t

    @cached_property
    def max_t(self):
        """Maximum time of the simulation (time of the snapshot) in [years]."""
        return self.snap.snap_time

    # ---------- HALO PROPERTIES ----------
    @cached_property
    def contamination_frac(self):
        """Contamination fraction: % of low resolution dark matter particles in the halo."""
        return self._load_or_compute_contamination_fraction()

    @cached_property
    def is_valid(self):
        """Check to see if the halo is valid, i.e., has a small contamination fraction (<1)."""

        # Max contamination fraction allowed
        max_contamination_frac = 1.0

        # Check if the halo is valid
        is_valid = self.contamination_frac <= max_contamination_frac

        return is_valid

    @cached_property
    def _has_enough_stars(self):
        """Check if the halo has enough stars (>=100)."""
        return len(self.pos_stars) >= 100

    # --- CM (center of mass) ---
    @cached_property
    def pos_cm(self):
        """Position of the center of mass of the halo in the simulation reference frame."""

        if self._has_enough_stars:
            return self._calculate_cm(self.mass_stars, self.pos_stars)

        # If we do not have many stars, we calculate the CM using dark matter
        else:
            return self._calculate_cm(self.mass_dm, self.pos_dm)

    @cached_property
    def vel_cm(self):
        """Velocity of the center of mass of the halo in the simulation reference frame."""

        if self._has_enough_stars:
            return self._calculate_cm(self.mass_stars, self.vel_stars)

        else:
            return self._calculate_cm(self.mass_dm, self.vel_dm)

    # --- ANGULAR MOMENTUM ---
    @cached_property
    def stars_angular_momentum(self):
        """Agular momentum of each individual star."""
        return self._compute_stars_angular_momentum()

    @cached_property
    def stars_angular_momentum_rotated(self):
        """Agular momentum of each individual star rotated so that z-axis is
        parallel to the global angular momentum of the halo."""

        if len(self.pos_stars) == 0:
            return np.array([0.0, 0.0, 0.0])

        return self.stars_angular_momentum @ self.rotation_matrix.T

    @cached_property
    def angular_momentum(self):
        """Angular momentum of the halo (using only stars)."""
        return self._compute_global_angular_momentum()

    @cached_property
    def v_rot_stars(self):
        """
        Rotational velocity of each star in the x-y plane (plane perpendicular
        to the total angular momentum).
        """
        return self._compute_v_rot_stars()

    # --- ROTATION MATRIX ---
    @cached_property
    def rotation_matrix(self):
        """Rotation matrix to rotate the coordinates from the simulation reference
        frame to the coordinate system where the z-axis is parallel to the global
        angular momentum.
        """
        return self._compute_rotation_matrix()

    # --- KINETIC ENERGY ---
    @cached_property
    def K_total(self):
        """Total kinetc energy of the halo (using only stars) in [M_sun * km^2 / s^2]."""
        if len(self.pos_stars) == 0:
            return 0.0

        vel_stars_scalar_squared = np.sum((self.vel_stars_cm) ** 2, axis=1)
        return np.sum(0.5 * self.mass_stars * vel_stars_scalar_squared)

    @cached_property
    def k_co(self):
        """Fraction of kinetic energy invested in ordered corotation (using only stars)."""
        return self._compute_k_co()

    @cached_property
    def is_spiral(self):
        """Check if the galaxy is spiral."""
        return self.k_co > 0.4

    # --- MASS ARRAY ---
    @cached_property
    def mass_g(self):
        """Array containing the masses of all the gas particles in [M_sun]."""
        return self._load_mass(0)

    @cached_property
    def mass_dm(self):
        """Array containing the masses of all the dark matter particles in [M_sun]."""
        return self._load_mass(1)

    @cached_property
    def mass_stars(self):
        """Array containing the masses of all the star particles in [M_sun]."""
        return self._load_mass(4)

    @cached_property
    def mass_bh(self):
        """Array containing the masses of all the black holes in [M_sun]."""
        return self._load_mass(5)

    # --- TOTAL MASS ---
    @cached_property
    def Mg(self):
        """Total mass of gas particles (sum of all the masses) in [M_sun]."""
        return self._compute_total_mass(self.mass_g)

    @cached_property
    def Mg_cold_sfr(self):
        """
        Total mass of cold gas particles in [M_sun] obtained form the SFR:
        adds only the mass of gas particles that are forming stars (that have
        SFR > 0).
        """

        mask_nonzero_sfr = self.sfr > 0
        mass_g_cold = self.mass_g[mask_nonzero_sfr]

        return self._compute_total_mass(mass_g_cold)

    @cached_property
    def Mg_cold_temperature(self):
        """
        Total mass of cold gas particles in [M_sun] obtained form the temperature:
        adds only the mass of gas particles that have T < 10^5 K.
        """

        mask = self.gas_temperature < 1e5
        mass_g_cold = self.mass_g[mask]

        return self._compute_total_mass(mass_g_cold)

    @cached_property
    def Mdm(self):
        """Total mass of dark matter particles (sum of all the masses) in [M_sun]."""
        return self._compute_total_mass(self.mass_dm)

    @cached_property
    def Mstars(self):
        """Total mass of star particles (sum of all the masses) in [M_sun]."""
        return self._compute_total_mass(self.mass_stars)

    @cached_property
    def Mbh(self):
        """Total mass of black holes (sum of all the masses) in [M_sun].
        The masses correspond to the internal masses of the blak holes."""
        return self._compute_total_mass(self.mass_bh)

    @cached_property
    def Mhalo(self):
        """Total mass of the halo (sum of all the masses corresponding to gas,
        dark matter, stars and black holes) in [M_sun]."""
        return self.Mg + self.Mdm + self.Mstars + self.Mbh

    # --- POSITIONS ---
    @cached_property
    def pos_g(self):
        """Positions of the gas particles in the simulation reference frame
        in [kpc]."""
        return self._load_position(0)

    @cached_property
    def pos_dm(self):
        """Positions of the dark matter particles in the simulation reference frame
        in [kpc]."""
        return self._load_position(1)

    @cached_property
    def pos_stars(self):
        """Positions of the star particles in the simulation reference frame
        in [kpc]."""
        return self._load_position(4)

    @cached_property
    def pos_bh(self):
        """Positions of the black holes in the simulation reference frame
        in [kpc]."""
        return self._load_position(5)

    # --- POSITIONS OF STARS (centered or rotated) ---
    # CENTERED POSITIONS (centered in the CM of the halo)
    @cached_property
    def pos_stars_cm(self):
        """Positions of the star particles in the center of mass reference frame
        in [kpc]."""
        if len(self.pos_stars) == 0:
            return self.pos_stars
        return self.pos_stars - self.pos_cm

    # Centered velocities
    @cached_property
    def vel_stars_cm(self):
        """Velocities of the star particles in the center of mass reference frame
        in [km/s]."""
        if len(self.vel_stars) == 0:
            return self.vel_stars
        return self.vel_stars - self.vel_cm

    # ROTATED POSITIONS (so z-axis is parellel to angular momentum of the halo)
    @cached_property
    def pos_stars_cm_rotated(self):
        """Positions of the star particles in the center of mass reference frame
        and rotated so the z-axis is parallel to the global angular momentum. In [kpc].
        """
        if len(self.pos_stars) == 0:
            return self.pos_stars
        return self._rotate_coords(self.pos_stars)

    # Distance (scalar) of each star to the CM projected on the x-y plane
    @cached_property
    def r_stars_projected(self):
        """Distance (scalar) of each star to the CM projected on the x-y plane in [kpc]."""
        if len(self.pos_stars) == 0:
            return 0.0
        return np.sqrt(
            self.pos_stars_cm_rotated[:, 0] ** 2 + self.pos_stars_cm_rotated[:, 1] ** 2
        )

    # --- POSITIONS DM CENTERED IN CM (radius) ---
    @cached_property
    def radius_dm(self):
        """Positions of the DM perticles in the center of mass reference frame,
        i.e., radius, in [kpc]."""
        pos_dm_cm = self.pos_dm - self.pos_cm
        return np.sqrt(np.sum(pos_dm_cm**2, axis=1))

    @cached_property
    def r_halo(self):
        """Radius of the halo, i.e., radius of the most distant DM particle,
        in [kpc]."""
        return np.max(self.radius_dm)

    # --- VELOCITIES ---
    @cached_property
    def vel_g(self):
        """Velocities of the gas particles in the simulation reference frame
        in [km/s]."""
        return self._load_vel(0)

    @cached_property
    def vel_dm(self):
        """Velocities of the dark matter particles in the simulation reference frame
        in [km/s]."""
        return self._load_vel(1)

    @cached_property
    def vel_stars(self):
        """Velocities of the star particles in the simulation reference frame
        in [km/s]."""
        return self._load_vel(4)

    @cached_property
    def vel_bh(self):
        """Velocities of the blak holes in the simulation reference frame
        in [km/s]."""
        return self._load_vel(5)

    # --- Gas Temperature ---
    @cached_property
    def gas_temperature(self):
        return self._load_gas_temperature()

    # --- SFR ---
    @cached_property
    def sfr(self):
        return self._load_gas_sfr()

    # --- SFT ---
    @cached_property
    def sft(self):
        return self._load_star_sft()

    # --- BHFT (Black Hole Formation Time) ---
    @cached_property
    def bhft(self):
        """BH Formation Time of the BHs belonging to the halo in [years]."""
        return self._load_bhft()

    # --- SFH ---
    @cached_property
    def sfh(self):
        """Star formation history of the halo."""
        return self._compute_sfh()

    # --- R50 ---
    @cached_property
    def r50(self):
        """Half-stellar-mass radius in [kpc]."""
        return self._compute_R50()

    # --- R1 ---
    @cached_property
    def r1(self):
        """Radius at which the stellar surface density equals 1M_sun*pc^-2 in [kpc]."""
        return self._compute_R1()

    # --- MAX ROTATION VELOCITY (for the Tully-Fisher relation) ---
    @cached_property
    def vmax(self):
        """Maximum rotation velocity of the galaxy (for the Tully-Fisher relation)."""
        return self._compute_vmax_from_Lz()

    # --- DM DENSITY PROFILE ---
    @cached_property
    def dm_density_profile(self):
        """DM density profile of the halo."""
        return self._compute_dm_density_profile()

    @cached_property
    def dm_inner_slope(self):
        """Inner slope alpha of the DM density profile [Di Cintio+2014]."""

        slope_01_Rvir_02 = self._compute_dm_inner_slope()
        slope_02_Rvir_03 = self._compute_dm_inner_slope(radial_range=(0.02, 0.03))
        slope_3_Softening_10 = self._compute_dm_inner_slope(
            radial_range=(3.0, 10.0), r_divided_by="softening"
        )
        slope_1_kpc_2 = self._compute_dm_inner_slope(
            radial_range=(1.0, 2.0), r_divided_by="kpc"
        )

        return np.array(
            [slope_01_Rvir_02, slope_02_Rvir_03, slope_3_Softening_10, slope_1_kpc_2]
        )

    # ------- END PROPERTIES -------

    def __read_AHF(self, f_AHF, halo_idx):

        with open(f_AHF, "r") as f:
            # Read the total number of halos (first line of the file)
            n_halos_total = int(f.readline().strip())
            # print(f"Total number of halos on the file: {n_halos_total}")

            if halo_idx > n_halos_total - 1:
                raise ValueError(
                    f"There are not enough halos. Requested halo: {halo_idx}, "
                    f"max halo index: {n_halos_total-1}"
                )

            print(f"Halo to be read: {halo_idx}")

            # NOTE: Since we already read the first line (n_halos_total),
            # this loop will start reading from the second line of the file.

            # Skip previous halos to reach the desired halo
            for _ in range(halo_idx):
                # Read the number of particles in this halo
                n_particles_prev = int(f.readline().split()[0])
                # Skip the particle lines of this halo
                for _ in range(n_particles_prev):
                    # f.readline()
                    next(f)

            # Now we are at the header of the desired halo: number of particles
            n_particles = int(f.readline().split()[0])
            print(f"Reading {n_particles} particles for halo {halo_idx} ...")

            if n_particles == 0:
                return 0, np.array([])

            # Allocate arrays for particle IDs and types
            ids = np.empty(n_particles, dtype=int)
            # types = np.empty(n_particles, dtype=int)

            # Read particle IDs and types
            for j in range(n_particles):
                pid, _ = map(int, f.readline().split())
                ids[j] = pid

        return n_particles, ids

    def _ids_in_halo(self, snapshot_part_ids, halo_part_ids):

        # This function returns the INDEXES (positions)
        # where the IDs of the halo match with the IDs of the snapshot

        # intersect1d with return_indices=True returns:
        # 1. Sorted array of common elements
        # 2. Indexes in the 1st array (snapshot) <--- THIS IS WHAT WE NEED
        # 3. Indexes in the 2nd array (halo)
        _, idx_in_snapshot, _ = np.intersect1d(
            snapshot_part_ids, halo_part_ids, assume_unique=True, return_indices=True
        )

        # h5py needs the indexes to be in ascending order, therefore we
        # need to sort them
        idx_in_snapshot_sorted = np.sort(idx_in_snapshot)

        return idx_in_snapshot_sorted

    def _load_ids(self, PartType):

        part_key = f"PartType{PartType}"
        ids = self.f[f"{part_key}/ParticleIDs"][:]

        return ids

    def _read_dataset(self, PartType, dataset_name, indices):
        """
        Auxiliar function to read the desired dataset in the HDF5 file
        """
        # If we do not have this particle type, return an empty array
        if len(indices) == 0:
            return np.array([])

        # Path to the dataset (ej: "PartType0/Coordinates")
        path = f"PartType{PartType}/{dataset_name}"

        # Read the file
        data = self.f[path][indices]

        return data

    def _load_position(self, PartType):

        pos_unit = self.f["Units"].attrs["KpcUnit"]

        # Select the indexes corresponding to the PartType (using the mapping
        # defined at the __init__)
        selected_indices = self.mapping[PartType]

        # Read the positions
        pos = self._read_dataset(PartType, "Coordinates", selected_indices)

        # If it is empty, return the empty array to avoid problems when multiplying
        if len(pos) == 0:
            return pos

        # Conversion to physical units: kpc
        pos *= pos_unit
        return pos

    def _load_mass(self, PartType):

        mass_unit = self.f["Units"].attrs["MsolUnit"]

        selected_indices = self.mapping[PartType]

        if PartType == 5:
            # If we want to read the BH masses, we read the internal mass
            mass = self._read_dataset(PartType, "InternalMass", selected_indices)

        else:
            # For the other PartTypes we read the normal masses
            mass = self._read_dataset(PartType, "Masses", selected_indices)

        if len(mass) == 0:
            return mass

        # Conversion to physical units: M_sun
        mass *= mass_unit

        return mass

    def _load_vel(self, PartType):

        vel_unit = self.f["Units"].attrs["KmPerSecUnit"]

        selected_indices = self.mapping[PartType]

        vel = self._read_dataset(PartType, "Velocities", selected_indices)

        if len(vel) == 0:
            return vel

        # Conversion to physical units: km/s
        vel *= vel_unit

        return vel

    def _load_star_sft(self):

        time_unit = self.f["Units"].attrs["SecUnit"]

        sft = self._read_dataset(4, "StellarFormationTime", self.idx_stars)

        if len(sft) == 0:
            return sft

        # Conversion to physical units: years
        # (31556926.0 sec = 1 year)
        sft *= time_unit / 31556926.0

        return sft

    def _load_gas_sfr(self):

        mass_unit = self.f["Units"].attrs["MsolUnit"]
        time_unit = self.f["Units"].attrs["SecUnit"]

        sfr = self._read_dataset(0, "StarFormationRate", self.idx_gas)

        if len(sfr) == 0:
            return sfr

        # Conversion to physical units: M_sun / year
        # (31556926.0 sec = 1 year)
        sfr *= mass_unit * 31556926.0 / time_unit

        return sfr

    def _load_bhft(self):

        time_unit = self.f["Units"].attrs["SecUnit"]

        bhft = self._read_dataset(5, "BHFormationTime", self.idx_bh)

        if len(bhft) == 0:
            return bhft

        # Conversion to physical units: years
        # (31556926.0 sec = 1 year)
        bhft *= time_unit / 31556926.0

        return bhft

    def _load_gas_density(self):

        density_unit = self.f["Units"].attrs["GmPerCcUnit"]

        rho = self._read_dataset(0, "Density", self.idx_gas)

        if len(rho) == 0:
            return rho

        # Conversion to physical units: g/cm^3
        rho *= density_unit

        return rho

    def _load_gas_temperature(self):

        temp = self._read_dataset(0, "Temperature", self.idx_gas)

        if len(temp) == 0:
            return temp

        # No need to convert to physical units because the temperature
        # is already in K (kelvin)

        return temp

    def _calculate_cm(self, mass, coords):

        return np.sum(coords * mass[:, np.newaxis], axis=0) / np.sum(mass)

    def _compute_stars_angular_momentum(self):

        if len(self.pos_stars) == 0:
            return np.array([0.0, 0.0, 0.0])

        # Linear and angular mom. of each individual star
        linear_mom = self.mass_stars[:, np.newaxis] * self.vel_stars_cm
        angular_mom = np.cross(self.pos_stars_cm, linear_mom)

        return angular_mom

    def _compute_global_angular_momentum(self):

        if len(self.pos_stars) == 0:
            return np.array([0.0, 0.0, 0.0])

        # Global angular momentum (of the halo)
        global_ang_mom = np.sum(self.stars_angular_momentum, axis=0)

        # Normalize angular momentum
        norm = np.linalg.norm(global_ang_mom)

        if norm == 0:
            return global_ang_mom

        return global_ang_mom / norm

    def _compute_rotation_matrix(self):

        # Norm angular momentum
        norm = np.linalg.norm(self.angular_momentum)

        if norm == 0:
            return np.identity(3)

        # Rotation angles
        gamma = np.arctan2(self.angular_momentum[1], self.angular_momentum[0])
        beta = np.arccos(self.angular_momentum[2])

        # Rotation matrix
        R = np.array(
            [
                [
                    np.cos(beta) * np.cos(gamma),
                    np.cos(beta) * np.sin(gamma),
                    -np.sin(beta),
                ],
                [-np.sin(gamma), np.cos(gamma), 0],
                [
                    np.sin(beta) * np.cos(gamma),
                    np.sin(beta) * np.sin(gamma),
                    np.cos(beta),
                ],
            ]
        )

        return R

    def _rotate_coords(self, original_coords):
        """
        Recieves coordinates, centers them in the CM and rotates them using the rotation matrix.
        The coordinates are rotated so the z-axis is parallel to the angular momentum of the halo.
        """
        if len(original_coords) == 0:
            return original_coords

        # Center positions in the CM
        centered_coords = original_coords - self.pos_cm

        # Rotate coords using the rotation matrix
        rotated_coords = centered_coords @ self.rotation_matrix.T

        return rotated_coords

    def _compute_total_mass(self, mass_array):
        if len(mass_array) == 0:
            return 0.0

        return np.sum(mass_array)

    def _compute_R50(self):

        # In case the halo has less than 100 stars, set R50 to zero
        if not self._has_enough_stars:
            return 0.0

        radius = np.sqrt(np.sum(self.pos_stars_cm**2, axis=1))

        valid_radii = radius[radius > 0]
        rmin = np.min(valid_radii)
        rmax = np.max(radius)

        bins_1d = np.logspace(np.log10(rmin), np.log10(rmax), 50)

        hist_mass, bin_edges = np.histogram(radius, bins_1d, weights=self.mass_stars)
        mass_cum = np.cumsum(hist_mass) / np.sum(hist_mass)

        r_axis = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        r50 = np.interp(0.5, mass_cum, r_axis)

        return r50

    '''
    def _compute_R50_m2(self):
        """
        More accurate than '_compute_R50', but more computationally expensive.
        """

        r2 = np.sum(self.pos_stars_cm**2, axis=1)
        idx = np.argsort(r2)

        r2_sorted = r2[idx]
        mass_sorted = self.mass_stars[idx]

        mass_cum = np.cumsum(mass_sorted)
        mass_cum /= mass_cum[-1]

        # r50_sq = r2_sorted[np.searchsorted(mass_cum, 0.5)]
        r50_sq = np.interp(0.5, mass_cum, r2_sorted)
        r50 = np.sqrt(r50_sq)

        return r50
    '''

    def _compute_stellar_surface_density_profile(self):
        """Compute the differencial face-on 2D stellar mass-density profile.
        Returns (Sigma, r_centers) in [M_sun / kpc^2] units.
        """

        if not self._has_enough_stars:
            return np.array([]), np.array([])

        # Radius of stars in the face-on plane (x-y plane)
        radius_face_on = self.r_stars_projected

        # Convert from kpc to pc
        # radius_face_on *= 1000.0

        valid_radii = radius_face_on[radius_face_on > 0]
        rmin = np.min(valid_radii)
        rmax = np.max(radius_face_on)

        bins = np.logspace(np.log10(rmin), np.log10(rmax), 100)

        # Mass on each ring
        hist_mass, bin_edges = np.histogram(
            radius_face_on, bins=bins, weights=self.mass_stars
        )

        # Area of each ring: pi * (R_outer^2 - R_inner^2)
        ring_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)

        # Surface density: M_sun / kpc^2
        Sigma = hist_mass / ring_areas

        # Bins centers (log scale)
        r_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Ignore empty bins
        mask = Sigma > 0
        return Sigma[mask], r_centers[mask]

    def _compute_R1(self):

        # In case the halo has less than 100 stars, set R1 to zero
        if not self._has_enough_stars:
            return 0.0

        # Compute face-on 2D stellar mass-density profile
        Sigma, r_axis = self._compute_stellar_surface_density_profile()

        # Find R1: radius at which the stellar mass density
        # profile = 1 Msun/pc^2 = 10^6 Msun/kpc^2

        def g(x):
            return np.interp(x, r_axis, Sigma) - 1e6

        try:
            sol = root_scalar(g, bracket=[np.min(r_axis), np.max(r_axis)])
            r1 = sol.root
            return r1

        except ValueError:
            # Is case the function does not change its sign, i.e., the surface
            # density does not go down below 1 M_sun/pc^2
            return np.nan

    # ------- TULLY-FISHER RELATION (rotation velocity) -------

    def _compute_k_co(self):

        if len(self.pos_stars) == 0:
            return 0.0

        # We consider only stars that follow the direction of rotation
        # of the galaxy, i.e., L_z,i is positive in the coord system in which
        # z-axis is parallel to the global angular momentum
        positive_mask = self.stars_angular_momentum_rotated[:, 2] > 0.0
        valid_mask = positive_mask & (self.r_stars_projected > 0)

        valid_L_zi = self.stars_angular_momentum_rotated[:, 2][valid_mask]

        if len(valid_L_zi) == 0.0:
            return 0.0

        valid_r = self.r_stars_projected[valid_mask]
        valid_mass = self.mass_stars[valid_mask]

        # Kinetic corotational energy
        # K_rot = Sum{ m_i/2 * (L_z,i / (m_i * R_i) )^2 },  Ref: Correa et al. 2017
        # where R_i is the distance of each star to the CM projected on the x-y plane
        # and with L_z,i positive
        K_rot = np.sum(0.5 * (valid_L_zi / valid_r) ** 2 / valid_mass)

        return K_rot / self.K_total

    def _compute_v_rot_stars(self):
        """
        Compute tangential (rotation) velocity of each star in the plane perpendicular
        to the total angular momentum, derived from the z-component of each
        star's angular momentum in the rotated frame.

        v_phi_i = L_z_i / (m_i * R_i)

        Returns array of shape (N_stars,) in [km/s].
        Positive values = corotating with the galaxy.
        """
        if len(self.pos_stars) == 0:
            return np.array([])

        # Select z-component of the angular momentum of each individual star
        # in the rotated frame (z-axis parallel to the total angular
        # momentum of the halo)
        L_z = self.stars_angular_momentum_rotated[:, 2]

        R = self.r_stars_projected  # projected radius in x-y plane [kpc]

        # Avoid division by zero for stars very close to the center
        safe_R = np.where(R > 0, R, np.nan)

        v_rot = L_z / (self.mass_stars * safe_R)

        return v_rot

    def _compute_vmax_from_Lz(self, n_bins=60, return_v_profile=False):
        """
        Computes V_max from the rotation velocity profile derived using L_z.
        Bins stars by projected radius and takes the mean v_phi per bin.

        If return_v_profile is True, appart from v_max, also returns
        the radius axis and the velocity profile: r, v
        """
        if len(self.pos_stars) == 0:
            return 0.0

        v_rot = self.v_rot_stars
        R = self.r_stars_projected

        # Keep only corotating stars (L_z > 0 in the coord system in which
        # z-axis is parallel to global angular momentum)
        mask = v_rot > 0
        R_co = R[mask]
        v_co = v_rot[mask]

        if len(R_co) == 0:
            return 0.0

        # Bin in log-space:
        # We leave out the stars that are closer than 1kpc to the CM of the halo
        # and cut at 50kpc (we could cut at 2*R1, but this way there is no need to
        # compute R1 just for this if R1 was not computed before)
        rmin = 1.0
        rmax = 50.0
        bins = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)

        # Sum of v in each bin and number of stars per bin
        v_sum, bin_edges = np.histogram(R_co, bins=bins, weights=v_co)
        counts, _ = np.histogram(R_co, bins=bins)

        # Mean v per bin (ignoring empty bins)
        nonempty = counts > 0
        v_mean = np.zeros(n_bins)
        v_mean[nonempty] = v_sum[nonempty] / counts[nonempty]

        if np.sum(nonempty) == 0:
            return 0.0

        if return_v_profile:
            r_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            return float(np.max(v_mean[nonempty])), r_centers, v_mean

        return float(np.max(v_mean[nonempty]))

    # -------- NFW PROFILE ---------

    def _compute_dm_density_profile(self, n_bins=25):
        """
        Computes the 3D spherical dark matter density profile.

        Bins DM particles by their 3D distance to the centre of mass using
        logarithmically spaced bins and computes the volume density in each
        shell: rho = M_shell / V_shell, where V_shell = (4/3)*pi*(r_outer^3 - r_inner^3).

        Returns
        -------
        rho : np.ndarray
            Dark matter density in each bin [M_sun / kpc^3].
        r_centers : np.ndarray
            Geometric centre of each radial bin [kpc].
        """
        if len(self.pos_dm) == 0:
            return np.array([]), np.array([])

        # 3D distances of DM particles from the centre of mass
        radius = self.radius_dm

        valid_radii = radius[radius > 0]
        if len(valid_radii) == 0:
            return np.array([]), np.array([])

        rmin = np.min(valid_radii)
        rmax = self.r_halo  # = np.max(self.radius_dm)

        bins = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)

        # Total DM mass in each spherical shell
        mass_in_shell, bin_edges = np.histogram(radius, bins=bins, weights=self.mass_dm)

        # Volume of each shell: (4/3)*pi*(r_outer^3 - r_inner^3)
        shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

        # Density [M_sun / kpc^3]
        rho = mass_in_shell / shell_volumes

        # Geometric bin centres (log scale)
        r_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])

        # Remove empty bins
        rho[rho == 0] = np.nan
        return rho, r_centers

    def _compute_dm_inner_slope(self, radial_range=(0.01, 0.02), r_divided_by="r_vir"):
        """
        Computes the inner slope alpha of the DM density profile by fitting a
        power law rho ∝ r^alpha in a given radial range, following the procedure
        described in Di Cintio et al. (2014).

        The fiducial range is 0.01 < r/R_vir < 0.02, but other ranges can be
        used to check robustness of the result.

        Parameters
        ----------
        radial_range : tuple, optional
            Inner and outer limits of the fitting range as fractions of r_vir.
            Default is (0.01, 0.02), i.e. 0.01 < r/R_vir < 0.02.
        n_bins : int, optional
            Number of logarithmic bins used to compute the density profile.

        Returns
        -------
        alpha : float
            Inner slope of the DM density profile. Returns np.nan if there are
            not enough points in the fitting range.
        """
        # rho, r_centers = self._compute_dm_density_profile(n_bins=n_bins)
        rho, r_centers = self.dm_density_profile

        if len(rho) == 0:
            return np.nan

        # Determine the scale factor based on r_divided_by
        if r_divided_by == "r_vir":
            scale = self.r_halo
        elif r_divided_by == "softening":
            scale = 0.5  # kpc
        elif r_divided_by == "kpc":
            scale = 1.0  # kpc
        else:
            raise ValueError(
                f"Unknown r_divided_by='{r_divided_by}'. "
                "Expected 'r_vir', 'softening', or 'kpc'."
            )

        r_inner = radial_range[0] * scale
        r_outer = radial_range[1] * scale

        # Select bins within the fitting range
        mask = (r_centers >= r_inner) & (r_centers <= r_outer)

        if np.sum(mask) < 2:
            print(
                f"Warning: fewer than 2 bins found in the range "
                f"[{r_inner:.3f}, {r_outer:.3f}] kpc. "
                f"Consider increasing n_bins or adjusting radial_range."
            )
            return np.nan

        log_r = np.log10(r_centers[mask])
        log_rho = np.log10(rho[mask])

        # Linear fit in log-log space: log(rho) = alpha * log(r) + const
        coeffs = np.polyfit(log_r, log_rho, deg=1)
        alpha = coeffs[0]

        return alpha

    # ------------------------------

    def _compute_sfh(self):

        # Verify if we have data in sft (maybe the halo doesn't have stars)
        if len(self.sft) == 0:
            # If there are no stars, return 99 zeros
            # (because we use 100 bins -> the histogram has 99 gaps)
            return np.zeros(99), np.zeros(99)

        # Time bins correspond to the hole time array of the simulation
        # this way we make sure that all halos have the same time axis
        time_bins = np.linspace(self.min_t, self.max_t, 100)
        bin_width = time_bins[1] - time_bins[0]  # in years

        # Masses per bin, not number of particles
        sfh, _ = np.histogram(self.sft, bins=time_bins, weights=self.mass_stars)

        # Divide for the bin's width to obtain M_sun/yr
        sfr = sfh / bin_width

        return sfr

    def _compute_contamination_fraction(self):
        low_resolution_dm_mask = self.mass_dm > self.min_dm_mass
        low_resolution_dm_array = self.mass_dm[low_resolution_dm_mask]

        total_len = len(self.mass_dm)

        if total_len == 0:
            return np.nan

        low_resolution_len = len(low_resolution_dm_array)
        contamination_fraction = low_resolution_len / total_len * 100.0

        return contamination_fraction

    def _load_or_compute_contamination_fraction(self):
        filename = os.path.join(self.output_dir, "Halo_ContaminationFraction.txt")

        if os.path.exists(filename):
            try:
                print("Trying to read contamination fraction file")
                data = np.loadtxt(filename, skiprows=1)  # skiprows=1 skips header
                # If the file has only one row, np.loadtxt returns a 1D array -> convert to 2D:
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                # Search the first row where the first column is equal to the halo idx
                mask = data[:, 0].astype(int) == self.idx
                if np.any(mask):
                    print(f"Contamination fraction read from file for halo {self.idx}.")
                    return float(data[mask, 1])
                else:
                    print(f"Halo {self.idx} not found in file. It will be computed.")
            except Exception as e:
                print(
                    f"Contamination fraction could not be read from file: {e}. It will be computed."
                )

        else:
            print("Contamination fraction file NOT found. It will be computed.")

        return self._compute_contamination_fraction()

    def _save_in_file(self, property_array, property_name):
        filename = os.path.join(self.output_dir, f"Halo_{property_name}.txt")

        file_exists = os.path.exists(filename)

        # Make sure that the property is a numpy array (even if it
        # is just one element)
        prop = np.atleast_1d(property_array)

        # Create one row containing all the data: [idx, val1, val2, ...]
        data_to_save = np.concatenate(([self.idx], prop)).reshape(1, -1)

        # Define the format dinamically
        # %d for the ID and %.16e for each element of the property
        formats = ["%d"] + ["%.16e"] * len(prop)

        with open(filename, "a") as f:
            np.savetxt(
                f,
                data_to_save,
                fmt=formats,
                header="" if file_exists else f"halo_id   {property_name}",
                comments="",
            )

    def save_contamination_fraction(self):
        # Always save the contamination fraction
        self._save_in_file(self.contamination_frac, "ContaminationFraction")

    def save_cm(self):
        # Always save the CM (even if the halo is contaminated)
        self._save_in_file(self.pos_cm, "CM")

    def save_Mg(self):
        # If the halo is valid, we save this property
        if self.is_valid:
            self._save_in_file(self.Mg, "Mg")

        # If the halo is not valid, we set this property to zero
        # without computing it
        else:
            self._save_in_file(0.0, "Mg")

    def save_Mg_cold_sfr(self):
        # If the halo is valid, we save this property
        if self.is_valid:
            self._save_in_file(self.Mg_cold_sfr, "Mg_cold_sfr")

        # If the halo is not valid, we set this property to zero
        # without computing it
        else:
            self._save_in_file(0.0, "Mg_cold_sfr")

    def save_Mg_cold_temperature(self):
        # If the halo is valid, we save this property
        if self.is_valid:
            self._save_in_file(self.Mg_cold_temperature, "Mg_cold_temperature")

        # If the halo is not valid, we set this property to zero
        # without computing it
        else:
            self._save_in_file(0.0, "Mg_cold_temperature")

    def save_Mdm(self):
        if self.is_valid:
            self._save_in_file(self.Mdm, "Mdm")
        else:
            self._save_in_file(0.0, "Mdm")

    def save_Mstars(self):
        if self.is_valid:
            self._save_in_file(self.Mstars, "Mstars")
        else:
            self._save_in_file(0.0, "Mstars")

    def save_Mbh(self):
        if self.is_valid:
            self._save_in_file(self.Mbh, "Mbh")
        else:
            self._save_in_file(0.0, "Mbh")

    def save_Mhalo(self):
        if self.is_valid:
            self._save_in_file(self.Mhalo, "Mhalo")
        else:
            self._save_in_file(0.0, "Mhalo")

    def save_sfr(self):
        if self.is_valid:
            self._save_in_file(self.sfr, "SFR")
        else:
            self._save_in_file(0.0, "SFR")

    def save_sfh(self):
        if self.is_valid:
            self._save_in_file(self.sfh, "SFH")
        else:
            self._save_in_file(np.zeros(99), "SFH")

    def save_r50(self):
        if self.is_valid:
            self._save_in_file(self.r50, "R50")
        else:
            self._save_in_file(0.0, "R50")

    def save_r1(self):
        if self.is_valid:
            self._save_in_file(self.r1, "R1")
        else:
            self._save_in_file(0.0, "R1")

    def save_k_co(self):
        if self.is_valid:
            self._save_in_file(self.k_co, "kco")
        else:
            self._save_in_file(0.0, "kco")

    def save_vmax(self):
        if self.is_valid and self.is_spiral:
            self._save_in_file(self.vmax, "Vmax")
        else:
            self._save_in_file(0.0, "Vmax")

    def save_r_halo(self):
        if self.is_valid:
            self._save_in_file(self.r_halo, "R200")
        else:
            self._save_in_file(0.0, "R200")

    def save_dm_inner_slope(self):

        filename = os.path.join(self.output_dir, "Halo_DM_InnerSlope.txt")
        file_exists = os.path.exists(filename)

        if self.is_valid:
            prop = np.atleast_1d(self.dm_inner_slope)
        else:
            prop = np.full(4, np.nan)

        data_to_save = np.concatenate(([self.idx], prop)).reshape(1, -1)
        formats = ["%d"] + ["%.16e"] * len(prop)

        header = (
            "halo_id   "
            "alpha_[0.01-0.02]Rvir   "
            "alpha_[0.02-0.03]Rvir   "
            "alpha_[3.0-10.0]soft(soft=0.5kpc)   "
            "alpha_[1.0-2.0]kpc"
        )

        with open(filename, "a") as f:
            np.savetxt(
                f,
                data_to_save,
                fmt=formats,
                header="" if file_exists else header,
                comments="",
            )

    def save_all_properties(self):

        self.save_contamination_fraction()
        self.save_cm()
        self.save_Mg()
        self.save_Mg_cold_sfr()
        self.save_Mg_cold_temperature()
        self.save_Mdm()
        self.save_Mstars()
        self.save_Mbh()
        self.save_Mhalo()
        self.save_sfr()
        self.save_sfh()
        self.save_r50()
        self.save_r1()
        self.save_k_co()
        self.save_vmax()
        self.save_r_halo()
        #self.save_dm_inner_slope()

    def plot_halo(
        self, PartTypes, plane=("x", "y"), styles=None, zoom=None, save_path=None
    ):

        # Positions map
        pos_map = {
            0: self.pos_g,
            1: self.pos_dm,
            4: self.pos_stars,
            5: self.pos_bh,
        }

        # If we only have one PartType, we convert it to a list to avoid
        # problems with the 'for' loop
        if isinstance(PartTypes, int):
            PartTypes = [PartTypes]

        # Default styles for each PartType
        default_styles = {
            0: dict(color="dodgerblue", marker=".", alpha=0.05, label="Gas"),
            1: dict(color="white", marker=".", alpha=0.03, label="DM"),
            4: dict(color="yellow", marker=".", alpha=0.2, label="Stars"),
            5: dict(color="red", alpha=1.0, s=50, label="BH"),
        }

        # If the user gives personalized styles, default styles are overwriten
        if styles is not None:
            for ptype, style in styles.items():
                default_styles[ptype].update(style)

        axis_map = {"x": 0, "y": 1, "z": 2}

        try:
            axis1 = axis_map[plane[0].lower()]
            axis2 = axis_map[plane[1].lower()]
        except (KeyError, IndexError, AttributeError):
            raise ValueError(
                "Incorrect axis values. 'plane' must contain two string elements "
                "from ('x', 'y', 'z')."
            )

        # NOTE: '.lower()' changes strings form capital letters to lower case letters
        # NOTE: 'KeyError': 'plane' elements are different from 'x', 'y', or 'z'.
        #       'IndexError': 'plane' does not have the expected number of elements: 2.
        #       'AttributeError': 'plane' elements (plane[i]) are not strings.

        # Black background
        fig, ax = plt.subplots()
        ax.set_facecolor("black")

        # Plot of each PartType
        for ptype in PartTypes:
            positions = pos_map[ptype]
            rotated_positions = self._rotate_coords(positions)

            ax.scatter(
                rotated_positions[:, axis1],
                rotated_positions[:, axis2],
                **default_styles[ptype],  # expands the dictionary as arguments
            )

        ax.set_aspect("equal")
        ax.set_xlabel(f"{plane[0]} [kpc]")
        ax.set_ylabel(f"{plane[1]} [kpc]")

        if zoom is not None:
            if "xlim" in zoom:
                ax.set_xlim(zoom["xlim"])
            if "ylim" in zoom:
                ax.set_ylim(zoom["ylim"])

        ax.legend()

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Particle distribution plot saved in: {save_path}")

        plt.show()

    def plot_stellar_density(
        self,
        plane=("x", "y"),
        n_bins=200,
        plot_bh=True,
        zoom=None,
        save_path=None,
        colormap="gist_gray",
        color_bh="red",
    ):
        axis_map = {"x": 0, "y": 1, "z": 2}
        try:
            axis1 = axis_map[plane[0].lower()]
            axis2 = axis_map[plane[1].lower()]
        except (KeyError, IndexError, AttributeError):
            raise ValueError(
                "Incorrect axis values. 'plane' must contain two string elements "
                "from ('x', 'y', 'z')."
            )

        rotated_stars_positions = self.pos_stars_cm_rotated

        if zoom is not None:
            xlim = zoom.get(
                "xlim",
                [
                    rotated_stars_positions[:, axis1].min(),
                    rotated_stars_positions[:, axis1].max(),
                ],
            )
            ylim = zoom.get(
                "ylim",
                [
                    rotated_stars_positions[:, axis2].min(),
                    rotated_stars_positions[:, axis2].max(),
                ],
            )
        else:
            total_max = np.max(
                [
                    rotated_stars_positions[:, axis1].max(),
                    rotated_stars_positions[:, axis2].max(),
                ]
            )
            total_min = np.min(
                [
                    rotated_stars_positions[:, axis1].min(),
                    rotated_stars_positions[:, axis2].min(),
                ]
            )
            xlim = [total_min, total_max]
            ylim = [total_min, total_max]

        mask = (
            (rotated_stars_positions[:, axis1] >= xlim[0])
            & (rotated_stars_positions[:, axis1] <= xlim[1])
            & (rotated_stars_positions[:, axis2] >= ylim[0])
            & (rotated_stars_positions[:, axis2] <= ylim[1])
        )
        filtered_pos = rotated_stars_positions[mask]
        filtered_mass = self.mass_stars[mask]

        bins_axis1 = np.linspace(xlim[0], xlim[1], n_bins)
        bins_axis2 = np.linspace(ylim[0], ylim[1], n_bins)
        bin_size1 = bins_axis1[1] - bins_axis1[0]
        bin_size2 = bins_axis2[1] - bins_axis2[0]
        stellar_density = filtered_mass / (bin_size1 * bin_size2)

        # --- Colormap with black empty bins ---
        cmap_chosen = plt.get_cmap(colormap).copy()
        cmap_chosen.set_bad("black")  # NaN → black
        # cmap_chosen.set_under("black") # valores bajo vmin → negro

        fig, ax = plt.subplots()
        ax.set_facecolor("black")  # black background

        im = ax.hist2d(
            filtered_pos[:, axis1],
            filtered_pos[:, axis2],
            bins=[bins_axis1, bins_axis2],
            norm="log",
            weights=stellar_density,
            cmap=cmap_chosen,
        )

        if plot_bh:
            rotated_bh_position = self._rotate_coords(self.pos_bh)
            ax.scatter(
                rotated_bh_position[:, axis1],
                rotated_bh_position[:, axis2],
                color=color_bh,
                alpha=1.0,
                s=50,
                label="BH",
            )

        fig.colorbar(
            im[3], ax=ax, label=r"Stellar surface density [$M\odot/\mathrm{kpc^{2}}$]"
        )
        ax.set_xlabel(f"{plane[0]} [kpc]")
        ax.set_ylabel(f"{plane[1]} [kpc]")
        # ax.legend(frameon=True)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Stellar density plot saved in: {save_path}")

        plt.show()

    def plot_gas_density(
        self, plane=("x", "y"), n_bins=200, plot_bh=True, zoom=None, save_path=None
    ):

        axis_map = {"x": 0, "y": 1, "z": 2}

        try:
            axis1 = axis_map[plane[0].lower()]
            axis2 = axis_map[plane[1].lower()]
        except (KeyError, IndexError, AttributeError):
            raise ValueError(
                "Incorrect axis values. 'plane' must contain two string elements "
                "from ('x', 'y', 'z')."
            )

        rotated_gas_positions = self._rotate_coords(self.pos_g)

        # Determine histogram limits
        if zoom is not None:
            xlim = zoom.get(
                "xlim",
                [
                    rotated_gas_positions[:, axis1].min(),
                    rotated_gas_positions[:, axis1].max(),
                ],
            )
            ylim = zoom.get(
                "ylim",
                [
                    rotated_gas_positions[:, axis2].min(),
                    rotated_gas_positions[:, axis2].max(),
                ],
            )
        else:
            total_max = np.max(
                [
                    rotated_gas_positions[:, axis1].max(),
                    rotated_gas_positions[:, axis2].max(),
                ]
            )
            total_min = np.min(
                [
                    rotated_gas_positions[:, axis1].min(),
                    rotated_gas_positions[:, axis2].min(),
                ]
            )
            xlim = [total_min, total_max]
            ylim = [total_min, total_max]

        # Filter particles before defining the bins
        mask = (
            (rotated_gas_positions[:, axis1] >= xlim[0])
            & (rotated_gas_positions[:, axis1] <= xlim[1])
            & (rotated_gas_positions[:, axis2] >= ylim[0])
            & (rotated_gas_positions[:, axis2] <= ylim[1])
        )
        filtered_pos = rotated_gas_positions[mask]
        filtered_mass = self.mass_g[mask]

        bins_axis1 = np.linspace(xlim[0], xlim[1], n_bins)
        bins_axis2 = np.linspace(ylim[0], ylim[1], n_bins)
        bin_size1 = bins_axis1[1] - bins_axis1[0]
        bin_size2 = bins_axis2[1] - bins_axis2[0]

        gas_density = filtered_mass / (bin_size1 * bin_size2)

        im = plt.hist2d(
            filtered_pos[:, axis1],
            filtered_pos[:, axis2],
            bins=[bins_axis1, bins_axis2],
            norm="log",
            weights=gas_density,
        )

        if plot_bh:
            rotated_bh_position = self._rotate_coords(self.pos_bh)
            plt.scatter(
                rotated_bh_position[:, axis1],
                rotated_bh_position[:, axis2],
                color="red",
                alpha=1.0,
                s=50,
                label="BH",
            )

        plt.colorbar(im[3], label=r"Gas surface density [$M_\odot/\mathrm{kpc^{2}}$]")
        plt.xlabel(f"{plane[0]} [kpc]")
        plt.ylabel(f"{plane[1]} [kpc]")
        plt.legend(frameon=True)

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Gas denisty plot saved in: {save_path}")

        plt.show()

    def plot_gas_temperature(
        self, plane=("x", "y"), n_bins=200, plot_bh=True, zoom=None, save_path=None
    ):

        axis_map = {"x": 0, "y": 1, "z": 2}

        try:
            axis1 = axis_map[plane[0].lower()]
            axis2 = axis_map[plane[1].lower()]
        except (KeyError, IndexError, AttributeError):
            raise ValueError(
                "Incorrect axis values. 'plane' must contain two string elements "
                "from ('x', 'y', 'z')."
            )

        rotated_gas_positions = self._rotate_coords(self.pos_g)
        temperature = self.gas_temperature
        mass = self.mass_g

        # Determine histogram limits
        if zoom is not None:
            xlim = zoom.get(
                "xlim",
                [
                    rotated_gas_positions[:, axis1].min(),
                    rotated_gas_positions[:, axis1].max(),
                ],
            )
            ylim = zoom.get(
                "ylim",
                [
                    rotated_gas_positions[:, axis2].min(),
                    rotated_gas_positions[:, axis2].max(),
                ],
            )
        else:
            total_max = np.max(
                [
                    rotated_gas_positions[:, axis1].max(),
                    rotated_gas_positions[:, axis2].max(),
                ]
            )
            total_min = np.min(
                [
                    rotated_gas_positions[:, axis1].min(),
                    rotated_gas_positions[:, axis2].min(),
                ]
            )
            xlim = [total_min, total_max]
            ylim = [total_min, total_max]

        # Filter particles before defining the bins
        mask = (
            (rotated_gas_positions[:, axis1] >= xlim[0])
            & (rotated_gas_positions[:, axis1] <= xlim[1])
            & (rotated_gas_positions[:, axis2] >= ylim[0])
            & (rotated_gas_positions[:, axis2] <= ylim[1])
        )
        x = rotated_gas_positions[mask, axis1]
        y = rotated_gas_positions[mask, axis2]
        filtered_mass = mass[mask]
        filtered_temp = temperature[mask]

        bins_x = np.linspace(xlim[0], xlim[1], n_bins)
        bins_y = np.linspace(ylim[0], ylim[1], n_bins)

        hist_mT, _, _ = np.histogram2d(
            x, y, bins=[bins_x, bins_y], weights=filtered_mass * filtered_temp
        )
        hist_m, _, _ = np.histogram2d(
            x, y, bins=[bins_x, bins_y], weights=filtered_mass
        )

        mean_T = np.divide(
            hist_mT, hist_m, out=np.zeros_like(hist_mT), where=hist_m > 0
        )

        plt.figure()

        im = plt.imshow(
            mean_T.T,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            norm=colors.LogNorm(),
            cmap="plasma",
        )

        if plot_bh:
            rotated_bh_position = self._rotate_coords(self.pos_bh)
            plt.scatter(
                rotated_bh_position[:, axis1],
                rotated_bh_position[:, axis2],
                color="red",
                s=50,
                label="BH",
            )

        plt.colorbar(im, label="Mass-weighted Temperature [K]")
        plt.xlabel(f"{plane[0]} [kpc]")
        plt.ylabel(f"{plane[1]} [kpc]")
        plt.gca().set_aspect("equal")
        plt.legend(frameon=True)

        if save_path is not None:
            plt.savefig(save_path)
            print(f"Gas temperature plot saved in: {save_path}")

        plt.show()
