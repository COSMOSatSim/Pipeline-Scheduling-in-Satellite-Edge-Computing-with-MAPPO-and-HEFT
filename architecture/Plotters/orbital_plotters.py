from skyfield.timelib import Time
import numpy as np
import matplotlib.pyplot as plt

    
def plot_orbital_corridor(sat_list, pos_A, pos_B, t:Time, corridor_angle_deg:float, show_footprints:bool=False):
    """
    Visualizza in 3D l'Observer (A), il Vessel (B) e i satelliti.
    Include la rotta curva sulla superficie e i 'pilastri' di altitudine dei satelliti.
    
    :param show_footprints: Se True, disegna i cerchi (cupole) a terra attorno ad A e B.
    :param corridor_angle_deg: Ampiezza del corridoio in gradi (usata per i footprint).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Coordinate e Costanti
    vA = pos_A.at(t).position.km
    vB = pos_B.at(t).position.km
    R_earth = 6371.0

    # 2. FUNZIONE INTERNA PER I FOOTPRINT (CERCHI)
    def draw_ground_circle(center_vec, angular_radius_deg, R, color, label):
        if angular_radius_deg <= 0: return
        
        rad = np.radians(angular_radius_deg)
        center_norm = center_vec / np.linalg.norm(center_vec)
        
        # Sistema di riferimento locale
        ref_vec = np.array([1, 0, 0]) if abs(center_norm[0]) < 0.9 else np.array([0, 1, 0])
        local_x = np.cross(center_norm, ref_vec)
        local_x /= np.linalg.norm(local_x)
        local_y = np.cross(center_norm, local_x)
        
        angles = np.linspace(0, 2*np.pi, 100)
        circle_pts = []
        for a in angles:
            # Generazione punto su calotta sferica
            pt_gc = (np.sin(rad) * np.cos(a) * local_x + 
                     np.sin(rad) * np.sin(a) * local_y + 
                     np.cos(rad) * center_norm) * R
            circle_pts.append(pt_gc)
        
        pts = np.array(circle_pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, 
                linestyle='-', linewidth=1.5, label=label, alpha=0.7)

    # 3. DISEGNO FOOTPRINT (Opzionale)
    if show_footprints:
        draw_ground_circle(vA, corridor_angle_deg, R_earth, 'green', 'Footprint Cupola A')
        draw_ground_circle(vB, corridor_angle_deg, R_earth, 'red', 'Footprint Cupola B')

    # 4. ROTTA CURVA SULLA SUPERFICIE
    t_vals = np.linspace(0, 1, 50)
    arc_points = []
    for val in t_vals:
        p = (1 - val) * vA + val * vB
        p_surf = (p / np.linalg.norm(p)) * R_earth
        arc_points.append(p_surf)
    arc_points = np.array(arc_points)
    
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 
            color='gray', linestyle='--', linewidth=2, label='Rotta Terrestre')

    # 5. PUNTI OBSERVER E VESSEL
    ax.scatter(*vA, color='green', s=100, label='Observer (A)', marker='o', depthshade=False, zorder=5)
    ax.scatter(*vB, color='red', s=100, label='Vessel (B)', marker='X', depthshade=False, zorder=5)

    # 6. SATELLITI E PILASTRI ALTITUDINE
    sat_coords = []
    if sat_list:
        sat_coords = np.array([sat.at(t).position.km for sat in sat_list])
        ax.scatter(sat_coords[:, 0], sat_coords[:, 1], sat_coords[:, 2], 
                   color='orange', s=50, label='Satelliti', marker='^', depthshade=False)

        for sat_pos in sat_coords:
            ground_pt = (sat_pos / np.linalg.norm(sat_pos)) * R_earth
            ax.plot([ground_pt[0], sat_pos[0]], [ground_pt[1], sat_pos[1]], [ground_pt[2], sat_pos[2]], 
                    color='orange', linestyle=':', alpha=0.6)

    # 7. SETUP ASSI E LEGENDA
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Vista 3D Corridoio Orbitale')
    
    # Rimuove duplicati in legenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # 8. PROPORZIONI REALI (Equi-spaced axes)
    all_p = np.vstack((arc_points, [vA, vB]))
    if len(sat_coords) > 0:
        all_p = np.vstack((all_p, sat_coords))
    
    mid = (all_p.max(axis=0) + all_p.min(axis=0)) * 0.5
    max_range = (all_p.max(axis=0) - all_p.min(axis=0)).max() / 2.0
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.show()