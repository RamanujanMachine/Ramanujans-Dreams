import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from pathlib import Path


def load_shards(filepath):
    """Loads the list of Shard objects from the pickle file."""
    path = Path(filepath)
    shards_list_l = []
    for file in path.glob("*.pkl"):
        with open(file, 'rb') as f:
            shards_list = pickle.load(f)
            shards_list_l.append(shards_list)
    print(f"Loaded {len(shards_list_l[0])} shards.")
    return shards_list_l[0]

def get_rotation_matrix(vec1, vec2):
    """ Aligns vec1 to vec2 using Rodrigues' rotation formula. """
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    c = np.dot(a, b)
    if c > 0.999999: return np.eye(3)
    if c < -0.999999:
        # True 180-degree rotation around an orthogonal axis (preserving right-hand rule)
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

#def get_rotation_matrix(vec1, vec2):
#     """ Aligns vec1 to vec2 using Rodrigues' rotation formula. """
#     a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
#     if s < 1e-8: return np.eye(3) if c > 0 else -np.eye(3)
#     kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
#     return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

def load_data_managers(folder_path):
    path = Path(folder_path)
    dfs = []
    for file in path.glob("*.pkl"):
        with open(file, 'rb') as f:
            dm = pickle.load(f)
            df = dm.as_df()
            df = df.dropna(subset=['delta'])
            print(df)

            if not df.empty and not len(df['sv']) < 10:
                dfs.append(df)
    print(f"Loaded {len(dfs)} DataManagers.")
    return dfs


def match_data_to_shards(dfs, shards):
    matched_pairs = []
    for df in dfs:
        start_dict = df.iloc[0]['sv'].start
        print(f'start at: {start_dict}')
        start_coords = list(v for _, v in start_dict.items())
        start_arr = np.array(start_coords, dtype=float)

        from dreamer.extraction.shard import Shard
        shard: Shard
        for shard in shards:
            shard_start_coords = list(v for _, v in shard.start_coord.items())
            shard_arr = np.array(shard_start_coords, dtype=float)

            if np.allclose(start_arr, shard_arr, atol=1e-5):
                matched_pairs.append((shard, df))
                break

    print(f"Successfully matched {len(matched_pairs)} Shard-Data pairs.")
    return matched_pairs


def plot_hyperplanes_on_sphere(ax, shard, cam_elev, cam_azim):
    try:
        A = np.array(shard.A, dtype=float)
    except Exception as e:
        print(f"Could not extract hyperplanes: {e}")
        return

    elev_rad, azim_rad = np.radians(cam_elev), np.radians(cam_azim)
    cam_vec = np.array([np.cos(elev_rad) * np.cos(azim_rad), np.cos(elev_rad) * np.sin(azim_rad), np.sin(elev_rad)])
    theta = np.linspace(0, 2 * np.pi, 300)

    for v in A:
        n = np.zeros(3);
        n[:len(v)] = v[:3]
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8: continue
        N = n / norm_n
        U = np.array([-N[1], N[0], 0.0]) if abs(N[0]) > 0.1 or abs(N[1]) > 0.1 else np.array([0.0, -N[2], N[1]])
        U /= np.linalg.norm(U)
        V = np.cross(N, U)
        circle_x, circle_y, circle_z = (1.001 * (np.cos(theta) * U[i] + np.sin(theta) * V[i]) for i in range(3))

        pts = np.vstack([circle_x, circle_y, circle_z]).T
        mask = pts.dot(cam_vec) < -0.1
        circle_x[mask], circle_y[mask], circle_z[mask] = np.nan, np.nan, np.nan
        ax.plot(circle_x, circle_y, circle_z, color='black', linewidth=1.5, alpha=0.85, zorder=5)
# def plot_hyperplanes_on_sphere(ax, shard, cam_elev, cam_azim):
#     try:
#         A = shard.A
#         print(f'cones: {shard.A} x < {shard.b}')
#         A = np.array(A, dtype=float)
#     except Exception as e:
#         print(f"Could not extract hyperplanes: {e}")
#         return
#
#     elev_rad = np.radians(cam_elev)
#     azim_rad = np.radians(cam_azim)
#     cam_vec = np.array([
#         np.cos(elev_rad) * np.cos(azim_rad),
#         np.cos(elev_rad) * np.sin(azim_rad),
#         np.sin(elev_rad)
#     ])
#
#     theta = np.linspace(0, 2 * np.pi, 300)
#
#     for v in A:
#         n = np.zeros(3)
#         n[:len(v)] = v[:3]
#         norm_n = np.linalg.norm(n)
#         if norm_n < 1e-8: continue
#
#         N = n / norm_n
#         R = 1.0
#
#         if abs(N[0]) > 0.1 or abs(N[1]) > 0.1:
#             U = np.array([-N[1], N[0], 0.0])
#         else:
#             U = np.array([0.0, -N[2], N[1]])
#         U = U / np.linalg.norm(U)
#         V = np.cross(N, U)
#
#         circle_x = R * np.cos(theta) * U[0] + R * np.sin(theta) * V[0]
#         circle_y = R * np.cos(theta) * U[1] + R * np.sin(theta) * V[1]
#         circle_z = R * np.cos(theta) * U[2] + R * np.sin(theta) * V[2]
#
#         circle_x *= 1.001
#         circle_y *= 1.001
#         circle_z *= 1.001
#
#         pts = np.vstack([circle_x, circle_y, circle_z]).T
#         dots = pts.dot(cam_vec)
#         mask = dots < -0.1
#
#         circle_x[mask] = np.nan
#         circle_y[mask] = np.nan
#         circle_z[mask] = np.nan
#
#         ax.plot(circle_x, circle_y, circle_z, color='black', linewidth=1.5, alpha=0.85, zorder=5)


# def draw_sphere_horizon(ax, cam_elev, cam_azim):
#     elev_rad = np.radians(cam_elev)
#     azim_rad = np.radians(cam_azim)
#     cam_vec = np.array([
#         np.cos(elev_rad) * np.cos(azim_rad),
#         np.cos(elev_rad) * np.sin(azim_rad),
#         np.sin(elev_rad)
#     ])
#
#     if abs(cam_vec[0]) > 0.1 or abs(cam_vec[1]) > 0.1:
#         u = np.array([-cam_vec[1], cam_vec[0], 0])
#     else:
#         u = np.array([0, -cam_vec[2], cam_vec[1]])
#     u = u / np.linalg.norm(u)
#     v = np.cross(cam_vec, u)
#
#     theta = np.linspace(0, 2 * np.pi, 200)
#     circle_x = np.cos(theta) * u[0] + np.sin(theta) * v[0]
#     circle_y = np.cos(theta) * u[1] + np.sin(theta) * v[1]
#     circle_z = np.cos(theta) * u[2] + np.sin(theta) * v[2]
#
#     ax.plot(circle_x * 1.001, circle_y * 1.001, circle_z * 1.001,
#             color='black', linewidth=1.5, alpha=0.8, zorder=2)

def draw_sphere_horizon(ax, cam_elev, cam_azim):
    elev_rad, azim_rad = np.radians(cam_elev), np.radians(cam_azim)
    cam_vec = np.array([np.cos(elev_rad) * np.cos(azim_rad), np.cos(elev_rad) * np.sin(azim_rad), np.sin(elev_rad)])
    u = np.array([-cam_vec[1], cam_vec[0], 0]) if abs(cam_vec[0]) > 0.1 or abs(cam_vec[1]) > 0.1 else np.array([0, -cam_vec[2], cam_vec[1]])
    u /= np.linalg.norm(u)
    v = np.cross(cam_vec, u)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.001*(np.cos(theta)*u[0]+np.sin(theta)*v[0]), 1.001*(np.cos(theta)*u[1]+np.sin(theta)*v[1]), 1.001*(np.cos(theta)*u[2]+np.sin(theta)*v[2]),
            color='black', linewidth=1.5, alpha=0.8, zorder=2)


from tqdm import tqdm

def generate_shard_atlas1(matched_pairs):
    n_shards = len(matched_pairs)
    if n_shards == 0:
        return

    cols = n_shards
    rows = 1

    plt.style.use('seaborn-v0_8-paper')
    # Reduced vertical height since horizontal colorbar is gone
    fig = plt.figure(figsize=(3.5 * cols + 1.5, 4.5), dpi=400)

    all_deltas = np.concatenate([df['delta'].dropna().values for _, df in matched_pairs])
    true_min = np.nanmin(all_deltas)
    true_max = np.nanmax(all_deltas)

    vmin_bound = np.floor(true_min / 0.2) * 0.2
    vmax_bound = np.ceil(true_max / 0.2) * 0.2

    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=vmin_bound, vmax=vmax_bound)

    for idx, (shard, df) in enumerate(matched_pairs):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.set_proj_type('ortho')

        try:
            ax.set_box_aspect((1, 1, 1), zoom=1.4)
        except TypeError:
            ax.set_box_aspect((1, 1, 1))

        df['delta'] = pd.to_numeric(df['delta'], errors='coerce')
        df = df.dropna(subset=['delta'])

        A_matrix = shard.A
        symbols = shard.symbols
        A_matrix = np.array(A_matrix, dtype=float)

        trajectories_list = []
        for _, row in df.iterrows():
            traj_dict = row['sv'].trajectory
            traj_vector = []
            for sym in symbols:
                val = next((v for k, v in traj_dict.items() if str(k) == str(sym)), 0.0)
                traj_vector.append(float(val))
            trajectories_list.append(traj_vector)

        trajectories = np.array(trajectories_list, dtype=float)
        deltas = df['delta'].values

        norms = np.linalg.norm(trajectories, axis=1, keepdims=True)
        norms[norms == 0] = 1
        unit_trajectories = trajectories / norms
        x, y = unit_trajectories[:, 0], unit_trajectories[:, 1]
        z = unit_trajectories[:, 2] if unit_trajectories.shape[1] >= 3 else np.zeros_like(x)

        best_idx = np.nanargmax(deltas)
        v_best = np.array([x[best_idx], y[best_idx], z[best_idx]])

        best_elev = np.degrees(np.arcsin(np.clip(v_best[2], -1.0, 1.0)))
        best_azim = np.degrees(np.arctan2(v_best[1], v_best[0]))

        v_target = np.array([1.0, 0.0, 0.0])
        R_mat = get_rotation_matrix(v_best, v_target)

        rotated_trajectories = unit_trajectories @ R_mat.T

        theta = np.arctan2(rotated_trajectories[:, 1], rotated_trajectories[:, 0])
        phi = np.arccos(np.clip(rotated_trajectories[:, 2], -1.0, 1.0))

        grid_res = 400j
        grid_theta, grid_phi = np.mgrid[np.min(theta):np.max(theta):grid_res, np.min(phi):np.max(phi):grid_res]
        grid_delta = griddata((theta, phi), deltas, (grid_theta, grid_phi), method='linear')

        grid_x_rot = np.cos(grid_theta) * np.sin(grid_phi)
        grid_y_rot = np.sin(grid_theta) * np.sin(grid_phi)
        grid_z_rot = np.cos(grid_phi)
        grid_pts_rot = np.c_[grid_x_rot.ravel(), grid_y_rot.ravel(), grid_z_rot.ravel()]

        grid_pts_orig = grid_pts_rot @ R_mat

        if A_matrix.shape[1] > grid_pts_orig.shape[1]:
            padding = np.zeros((grid_pts_orig.shape[0], A_matrix.shape[1] - grid_pts_orig.shape[1]))
            grid_pts_padded = np.hstack((grid_pts_orig, padding))
        else:
            grid_pts_padded = grid_pts_orig[:, :A_matrix.shape[1]]

        boundary_checks = grid_pts_padded @ A_matrix.T
        outside_math_mask = np.any(boundary_checks > 1e-6, axis=1)
        grid_delta.ravel()[outside_math_mask] = np.nan

        tree = cKDTree(unit_trajectories)
        distances, _ = tree.query(grid_pts_orig)
        grid_delta.ravel()[distances > 0.2] = np.nan

        grid_x = grid_pts_orig[:, 0].reshape(grid_theta.shape)
        grid_y = grid_pts_orig[:, 1].reshape(grid_theta.shape)
        grid_z = grid_pts_orig[:, 2].reshape(grid_theta.shape)

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='white', alpha=0, edgecolor='none')
        # ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.1, linewidth=0.5) # 0.3
        draw_sphere_horizon(ax, best_elev, best_azim)

        colors = cmap(norm(grid_delta))
        colors[np.isnan(grid_delta), 3] = 0.0

        ax.plot_surface(grid_x, grid_y, grid_z, facecolors=colors, shade=False,
                        edgecolor='none', antialiased=True)

        plot_hyperplanes_on_sphere(ax, shard, best_elev, best_azim)

        ax.view_init(elev=best_elev, azim=best_azim)
        ax.axis('off')

        ax_inset = ax.inset_axes([-0.1, -0.1, 0.25, 0.25], projection='3d')
        ax_inset.set_proj_type('ortho')
        ax_inset.axis('off')
        l = 1.0

        ax_inset.scatter([0], [0], [0], color='black', s=15, zorder=5)

        ax_inset.quiver(0, 0, 0, l, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=1.5, zorder=4)
        ax_inset.quiver(0, 0, 0, 0, l, 0, color='g', arrow_length_ratio=0.2, linewidth=1.5, zorder=4)
        ax_inset.quiver(0, 0, 0, 0, 0, l, color='b', arrow_length_ratio=0.2, linewidth=1.5, zorder=4)
        ax_inset.text(l * 1.2, 0, 0, 'X', color='r', fontsize=10, weight='bold')
        ax_inset.text(0, l * 1.2, 0, 'Y', color='g', fontsize=10, weight='bold')
        ax_inset.text(0, 0, l * 1.2, 'Z', color='b', fontsize=10, weight='bold')
        ax_inset.set_xlim([0, l])
        ax_inset.set_ylim([0, l])
        ax_inset.set_zlim([0, l])
        ax_inset.view_init(elev=best_elev, azim=best_azim)

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    # --- MOVED COLORBAR TO THE RIGHT SIDE ---
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.88, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')

    ticks = np.arange(vmin_bound, vmax_bound + 0.01, 0.2)
    cbar.set_ticks(ticks)
    # Using set_yticklabels for vertical bar
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=13)
    cbar.set_label(r'Irrationality Measure ($\delta$)', fontsize=16, labelpad=15)

    # Make room for the right-side colorbar (right=0.88) and the bottom text (bottom=0.15)
    plt.subplots_adjust(left=0.02, right=0.88, top=0.98, bottom=0.15, wspace=-0.05)
    plt.show()


def generate_shard_atlas2(matched_pairs):
    n_shards = len(matched_pairs)

    # ── Figure: spheres slightly shorter than colorbar ────────────────────
    fig = plt.figure(figsize=(4.2 * n_shards + 1.5, 4.2), dpi=400)

    all_deltas = np.concatenate([df['delta'].dropna().values for _, df in matched_pairs])
    norm = plt.Normalize(vmin=np.floor(np.nanmin(all_deltas) / 0.2) * 0.2,
                         vmax=np.ceil(np.nanmax(all_deltas) / 0.2) * 0.2)
    cmap = plt.get_cmap('coolwarm')

    for idx, (shard, df) in tqdm(enumerate(matched_pairs)):
        ax = fig.add_subplot(1, n_shards, idx + 1, projection='3d')
        ax.set_proj_type('persp')

        ax.set_box_aspect((1, 1, 1), zoom=1.32)

        # 1. Trajectory Extraction
        trajectories = np.array([[float(next((v for k, v in row['sv'].trajectory.items() if str(k) == str(sym)), 0.0))
                                  for sym in shard.symbols] for _, row in df.iterrows()])
        unit_trajectories = trajectories / np.linalg.norm(trajectories, axis=1, keepdims=True)

        # 2. Local Equator Interpolation (Resolution Upgrade)
        best_idx = np.nanargmax(df['delta'].values)
        v_best = unit_trajectories[best_idx, :3]
        R_to_eq = get_rotation_matrix(v_best, np.array([1.0, 0.0, 0.0]))
        rotated_pts = unit_trajectories[:, :3] @ R_to_eq.T

        dt, dp = np.arctan2(rotated_pts[:, 1], rotated_pts[:, 0]), np.arccos(np.clip(rotated_pts[:, 2], -1, 1))
        grid_theta, grid_phi = np.mgrid[np.min(dt) - 0.1:np.max(dt) + 0.1:400j, np.min(dp) - 0.1:np.max(dp) + 0.1:400j]

        grid_delta = griddata((dt, dp), df['delta'].values, (grid_theta, grid_phi), method='linear')
        grid_delta = np.where(np.isnan(grid_delta),
                              griddata((dt, dp), df['delta'].values, (grid_theta, grid_phi), method='nearest'),
                              grid_delta)

        # 3. Restore to Upright Global Sphere
        gx, gy, gz = np.cos(grid_theta) * np.sin(grid_phi), np.sin(grid_theta) * np.sin(grid_phi), np.cos(grid_phi)
        grid_pts_orig = np.c_[gx.ravel(), gy.ravel(), gz.ravel()] @ R_to_eq

        # 4. Math Boundary Trimming
        A_float = np.array(shard.A, dtype=float)
        A_norm = A_float / np.linalg.norm(A_float, axis=1, keepdims=True)
        grid_delta.ravel()[np.any(grid_pts_orig[:, :A_norm.shape[1]] @ A_norm.T > 1e-4, axis=1)] = np.nan
        grid_delta.ravel()[cKDTree(unit_trajectories[:, :3]).query(grid_pts_orig)[0] > 0.18] = np.nan

        # 5. Rendering
        colors = cmap(norm(grid_delta))
        colors[np.isnan(grid_delta), 3] = 0.0
        sx, sy, sz = (grid_pts_orig[:, i].reshape(400, 400) for i in range(3))
        ax.plot_surface(sx, sy, sz, facecolors=colors, shade=False, antialiased=True, rcount=400, ccount=400, zorder=5)

        # 6. Static grid via plot_surface — poles fixed at z=±1, equator at z=0.
        #    World z is always "up" regardless of camera. 10° spacing.
        u_wf = np.linspace(0, 2 * np.pi, 37)  # 36 meridians → every 10°
        v_wf = np.linspace(0, np.pi, 19)  # 18 parallels  → every 10°
        ax.plot_surface(
            np.outer(np.cos(u_wf), np.sin(v_wf)),
            np.outer(np.sin(u_wf), np.sin(v_wf)),
            np.outer(np.ones_like(u_wf), np.cos(v_wf)),
            color='white', alpha=0.0,
            edgecolor='gray', linewidth=0.35, shade=False, zorder=1
        )

        elev, azim = np.degrees(np.arcsin(v_best[2])), np.degrees(np.arctan2(v_best[1], v_best[0]))
        draw_sphere_horizon(ax, elev, azim)
        plot_hyperplanes_on_sphere(ax, shard, elev, azim)

        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')

    # ── Layout: slightly negative wspace compensates for the large empty
    #    padding that matplotlib adds around 3D subplot boxes; gives ~1 cm
    #    visual gap between the sphere outlines. ────────────────────────────
    plt.subplots_adjust(left=0.01, right=0.85, top=0.97, bottom=0.03, wspace=-0.15)

    cbar_ax = fig.add_axes([0.87, 0.12, 0.018, 0.76])  # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)

    vmin_b = np.floor(np.nanmin(all_deltas) / 0.2) * 0.2
    vmax_b = np.ceil(np.nanmax(all_deltas) / 0.2) * 0.2
    ticks = np.arange(vmin_b, vmax_b + 0.01, 0.2)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=13)
    cbar.set_label(r'Irrationality Measure ($\delta$)', fontsize=16, labelpad=15)

    plt.show()


if __name__ == "__main__":
    DATA_FOLDER_log = "../examples/search results/log-2/pFq_2_1_-1__0_0_0__log-2"  # Folder with your DataManager .pickle files
    SHARDS_FILE_log = "../examples/spaces/log-2/pFq_2_1_-1__0_0_0__log-2"  # File containing the list of Shard objects
    DATA_FOLDER_pi = "../examples/search results/pi/zz_pi__0_0_0__pi"  # Folder with your DataManager .pickle files
    SHARDS_FILE_pi = "../examples/spaces/pi/zz_pi__0_0_0__pi"  # File containing the list of Shard objects


    shards_list = load_shards(SHARDS_FILE_log)
    df_list = load_data_managers(DATA_FOLDER_log)
    df_list = [df_list[2], df_list[1]]
    pairs = match_data_to_shards(df_list, shards_list)
    generate_shard_atlas2(pairs)

    shards_list = load_shards(SHARDS_FILE_pi)
    for shard in shards_list:
        print(shard.start_coord)
    df_list = load_data_managers(DATA_FOLDER_pi)
    df_list = [df_list[0], df_list[-1]]
    pairs = match_data_to_shards(df_list, shards_list)
    generate_shard_atlas2(pairs)