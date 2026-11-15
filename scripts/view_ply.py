#!/usr/bin/env python3
"""Lightweight interactive .ply mesh & camera viewer using plotly."""

import argparse
from pathlib import Path
import plotly.graph_objects as go


def load_ply_ascii(path: Path):
    xs, ys, zs = [], [], []
    colors = []
    cam_xs, cam_ys, cam_zs = [], [], []
    fi, fj, fk = [], [], []

    n_verts = 0
    n_faces = 0
    in_header = True
    vert_idx = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if in_header:
                if s.startswith("element vertex "):
                    n_verts = int(s.split()[2])
                elif s.startswith("element face "):
                    n_faces = int(s.split()[2])
                elif s == "end_header":
                    in_header = False
                continue
            if not s:
                continue
            parts = s.split()
            if vert_idx < n_verts:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                if (r, g, b) == (0, 255, 0):
                    cam_xs.append(x)
                    cam_ys.append(y)
                    cam_zs.append(z)
                else:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    colors.append(f"rgb({r},{g},{b})")
                vert_idx += 1
            elif len(fi) < n_faces:
                # Format: 3 i j k
                fi.append(int(parts[1]))
                fj.append(int(parts[2]))
                fk.append(int(parts[3]))

    return (xs, ys, zs, colors), (fi, fj, fk), (cam_xs, cam_ys, cam_zs)


def main():
    parser = argparse.ArgumentParser(description="View ASCII .ply mesh/cloud reconstruction as HTML.")
    parser.add_argument("ply", type=Path, help="Path to .ply file")
    parser.add_argument("--out", type=Path, default=Path("output/view.html"), help="Output HTML path")
    args = parser.parse_args()

    (xs, ys, zs, colors), (fi, fj, fk), (cxs, cys, czs) = load_ply_ascii(args.ply)

    fig = go.Figure()
    if fi:
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=fi,
                j=fj,
                k=fk,
                vertexcolor=colors,
                name="Colored Mesh",
                flatshading=False,
                lighting=dict(ambient=0.85, diffuse=0.5, specular=0.1),
            )
        )
        title = f"SfM Colored Mesh ({len(xs)} vertices, {len(fi)} triangles, {len(cxs)} cameras)"
    else:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name="3D Points",
                marker=dict(size=3.0, color=colors),
            )
        )
        title = f"SfM Point Cloud ({len(xs)} points, {len(cxs)} cameras)"

    if cxs:
        fig.add_trace(
            go.Scatter3d(
                x=cxs,
                y=cys,
                z=czs,
                mode="markers+lines",
                name="Cameras",
                marker=dict(size=6, color="lime", symbol="diamond"),
                line=dict(color="lime", width=2),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), auto_open=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
