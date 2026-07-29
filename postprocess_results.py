import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# RESULT POSTPROCESSING
# ------------------------------------------------------------
# Reads a saved solver NPZ and generates field plots only when
# requested. The solver itself should remain focused on solving.
# ============================================================


def load_npz_as_dict(path):
    data = np.load(path, allow_pickle=True)
    out = {}
    for key in data.files:
        value = data[key]
        if getattr(value, "dtype", None) == object and value.shape == ():
            out[key] = value.item()
        else:
            out[key] = value
    return out


def get_field(data, primary, fallback=None):
    if primary in data:
        return np.asarray(data[primary], dtype=float)
    if fallback is not None and fallback in data:
        return np.asarray(data[fallback], dtype=float)
    raise KeyError(f"Could not find field '{primary}'" + (f" or '{fallback}'" if fallback else ""))


def build_masks(region, region_defs):
    is_fluid = np.zeros_like(region, dtype=bool)
    is_solid = np.zeros_like(region, dtype=bool)
    for region_id, info in region_defs.items():
        phase = str(info.get("phase", "")).lower()
        mask = region == int(region_id)
        if phase == "fluid":
            is_fluid |= mask
        elif phase == "solid":
            is_solid |= mask
    return {"fluid": is_fluid, "solid": is_solid}


def get_coordinates(data):
    if "x" in data and "y" in data:
        return np.asarray(data["x"], dtype=float), np.asarray(data["y"], dtype=float)

    nx = int(data["nx"])
    ny = int(data["ny"])
    Lx = float(data["Lx"])
    Ly = float(data["Ly"])
    dx = Lx / nx
    dy = Ly / ny
    x = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    y = np.linspace(dy / 2.0, Ly - dy / 2.0, ny)
    return x, y


def masked_field(field, is_solid):
    return np.ma.array(field, mask=is_solid)


def apply_physical_axes(Lx, Ly):
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(0.0, Lx)
    plt.ylim(0.0, Ly)


def save_contour(out_dir, X, Y, field, is_solid, Lx, Ly, title, cbar_label, filename):
    fig = plt.figure(figsize=(8, 4.5))
    f = masked_field(field, is_solid)
    plt.pcolormesh(X, Y, f, shading="nearest")
    plt.colorbar(label=cbar_label)

    obstacle = np.ma.masked_where(~is_solid, is_solid.astype(float))
    plt.pcolormesh(X, Y, obstacle, shading="nearest", cmap="gray", alpha=0.35)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    apply_physical_axes(Lx, Ly)
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300)
    plt.close(fig)


def save_temperature_contour(out_dir, X, Y, T, is_solid, Lx, Ly, filename):
    fig = plt.figure(figsize=(8, 4.5))
    plt.pcolormesh(X, Y, T, shading="nearest")
    plt.colorbar(label="T")

    obstacle_outline = np.ma.masked_where(~is_solid, is_solid.astype(float))
    if np.any(is_solid):
        plt.contour(X, Y, obstacle_outline, levels=[0.5], colors="gray", linewidths=1.2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("temperature")
    apply_physical_axes(Lx, Ly)
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300)
    plt.close(fig)


def save_velocity_magnitude(out_dir, X, Y, u, v, is_solid, Lx, Ly, filename):
    speed = np.sqrt(masked_field(u, is_solid).filled(np.nan) ** 2 + masked_field(v, is_solid).filled(np.nan) ** 2)
    fig = plt.figure(figsize=(8, 4.5))
    plt.pcolormesh(X, Y, np.ma.masked_invalid(speed), shading="nearest")
    plt.colorbar(label="|V|")

    obstacle = np.ma.masked_where(~is_solid, is_solid.astype(float))
    plt.pcolormesh(X, Y, obstacle, shading="nearest", cmap="gray", alpha=0.35)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity magnitude")
    apply_physical_axes(Lx, Ly)
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300)
    plt.close(fig)


def save_vector_field(out_dir, X, Y, u, v, is_solid, Lx, Ly, filename, target_vectors_x=45, target_vectors_y=20):
    ny, nx = u.shape
    uu = masked_field(u, is_solid).filled(np.nan)
    vv = masked_field(v, is_solid).filled(np.nan)
    speed = np.sqrt(uu ** 2 + vv ** 2)

    stride_x = max(1, nx // target_vectors_x)
    stride_y = max(1, ny // target_vectors_y)

    Xq = X[::stride_y, ::stride_x]
    Yq = Y[::stride_y, ::stride_x]
    Uq = uu[::stride_y, ::stride_x]
    Vq = vv[::stride_y, ::stride_x]
    Sq = speed[::stride_y, ::stride_x]

    fig = plt.figure(figsize=(8, 4.5))
    bg = plt.pcolormesh(X, Y, np.ma.masked_invalid(speed), shading="nearest", alpha=0.35)
    plt.colorbar(bg, label="|V|")

    obstacle = np.ma.masked_where(~is_solid, is_solid.astype(float))
    plt.pcolormesh(X, Y, obstacle, shading="nearest", cmap="gray", alpha=0.35)

    finite_mask = np.isfinite(Uq) & np.isfinite(Vq) & np.isfinite(Sq)
    if np.any(finite_mask) and (np.nanmax(np.abs(Uq)) > 0.0 or np.nanmax(np.abs(Vq)) > 0.0):
        plt.quiver(
            Xq[finite_mask], Yq[finite_mask], Uq[finite_mask], Vq[finite_mask], Sq[finite_mask],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=None,
            pivot="mid",
            width=0.0020,
            headwidth=3.8,
            headlength=5.0,
            headaxislength=4.5,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity vectors over speed field")
    apply_physical_axes(Lx, Ly)
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_streamlines(out_dir, X, Y, u, v, is_solid, Lx, Ly, filename):
    uu = masked_field(u, is_solid).filled(np.nan)
    vv = masked_field(v, is_solid).filled(np.nan)
    speed = np.sqrt(uu ** 2 + vv ** 2)

    fig = plt.figure(figsize=(8, 4.5))
    cf = plt.contourf(X, Y, np.ma.masked_invalid(speed), levels=30, alpha=0.35)
    plt.colorbar(cf, label="|V|")

    obstacle = np.ma.masked_where(~is_solid, is_solid.astype(float))
    plt.pcolormesh(X, Y, obstacle, shading="nearest", cmap="gray", alpha=0.45)

    uu_stream = np.nan_to_num(uu, nan=0.0)
    vv_stream = np.nan_to_num(vv, nan=0.0)
    if np.max(np.abs(uu_stream)) > 0.0 or np.max(np.abs(vv_stream)) > 0.0:
        plt.streamplot(X, Y, uu_stream, vv_stream, density=1.4, linewidth=1.0)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Streamlines and speed")
    apply_physical_axes(Lx, Ly)
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_centerline_plots(out_dir, x, y, u, v, p, filename_prefix="centerline"):
    ny, nx = u.shape
    i_mid = nx // 2
    j_mid = ny // 2

    fig = plt.figure(figsize=(8, 5))
    plt.plot(u[:, i_mid], y, label="u(x=Lx/2, y)")
    plt.plot(v[:, i_mid], y, label="v(x=Lx/2, y)")
    plt.xlabel("Velocity")
    plt.ylabel("y")
    plt.title("Vertical centerline velocity profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_vertical_velocity.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, u[j_mid, :], label="u(x, y=Ly/2)")
    plt.plot(x, v[j_mid, :], label="v(x, y=Ly/2)")
    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.title("Horizontal centerline velocity profiles")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_horizontal_velocity.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(p[:, i_mid], y, label="p(x=Lx/2, y)")
    plt.xlabel("Pressure")
    plt.ylabel("y")
    plt.title("Vertical centerline pressure profile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_vertical_pressure.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, p[j_mid, :], label="p(x, y=Ly/2)")
    plt.xlabel("x")
    plt.ylabel("Pressure")
    plt.title("Horizontal centerline pressure profile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{filename_prefix}_horizontal_pressure.png", dpi=300)
    plt.close(fig)


def postprocess_result(npz_file, output_dir=None):
    npz_path = Path(npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"Result file not found: {npz_path}")

    data = load_npz_as_dict(npz_path)
    u = get_field(data, "solved_u", "u")
    v = get_field(data, "solved_v", "v")
    p = get_field(data, "solved_p", "p")
    T = get_field(data, "solved_T", "T")

    region = np.asarray(data["region"])
    region_defs = data["region_defs"]
    masks = build_masks(region, region_defs)
    is_solid = masks["solid"]

    x, y = get_coordinates(data)
    X, Y = np.meshgrid(x, y)
    Lx = float(data["Lx"])
    Ly = float(data["Ly"])

    out_dir = Path(output_dir) if output_dir is not None else npz_path.parent / f"{npz_path.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_contour(out_dir, X, Y, u, is_solid, Lx, Ly, "u velocity", "u", "u_velocity.png")
    save_contour(out_dir, X, Y, v, is_solid, Lx, Ly, "v velocity", "v", "v_velocity.png")
    save_contour(out_dir, X, Y, p, is_solid, Lx, Ly, "pressure", "p", "pressure.png")
    save_temperature_contour(out_dir, X, Y, T, is_solid, Lx, Ly, "temperature.png")
    save_velocity_magnitude(out_dir, X, Y, u, v, is_solid, Lx, Ly, "velocity_magnitude.png")
    save_vector_field(out_dir, X, Y, u, v, is_solid, Lx, Ly, "velocity_vectors.png")
    save_streamlines(out_dir, X, Y, u, v, is_solid, Lx, Ly, "streamlines.png")
    save_centerline_plots(out_dir, x, y, u, v, p, filename_prefix="centerline")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate plots from a saved CFD result NPZ.")
    parser.add_argument("npz_file", help="Path to result NPZ file produced by the solver.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for generated plots.")
    args = parser.parse_args()

    out_dir = postprocess_result(args.npz_file, args.output_dir)
    print(f"Post-processing plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
