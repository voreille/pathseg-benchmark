from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import click
import numpy as np
import openslide
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from pathseg.models.encoder import build_encoder
from histoseg_plugin.core.tissue.segmentation import segment_tissue
from histoseg_plugin.core.tiling.tile import generate_tiles_from_tissue


class TileDataset(Dataset):
    def __init__(
        self,
        wsi: openslide.OpenSlide,
        coords: np.ndarray,
        tile_size: int,
        tile_level: int,
        transforms: Optional[Any] = None,
    ):
        self.wsi = wsi
        self.coords = np.asarray(coords)
        self.tile_size = int(tile_size)
        self.tile_level = int(tile_level)
        self.transforms = transforms or T.ToTensor()

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        x0, y0 = self.coords[idx]

        tile = self.wsi.read_region(
            (int(x0), int(y0)),
            self.tile_level,
            (self.tile_size, self.tile_size),
        ).convert("RGB")

        return self.transforms(tile), torch.tensor([x0, y0], dtype=torch.long)


def build_tile_transform(px_mean: Sequence[float], px_std: Sequence[float]):
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=px_mean, std=px_std),
        ]
    )


def tile_slide(
    wsi: openslide.OpenSlide,
    tile_level: int,
    tile_size: int,
    step_size: int,
) -> np.ndarray:
    contours, holes = segment_tissue(wsi)

    coords = generate_tiles_from_tissue(
        wsi,
        contours,
        holes,
        tile_level=tile_level,
        tile_size=tile_size,
        step_size=step_size,
        max_workers=1,
    )

    return np.asarray(coords)


def compute_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings = []
    all_coords = []

    model.eval()

    for tiles, batch_coords in tqdm(dataloader, desc="Computing embeddings"):
        tiles = tiles.to(device, non_blocking=True)

        with torch.no_grad():
            if use_amp and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(tiles)
            else:
                output = model(tiles)

        # ViT-like encoder output: take CLS token.
        # If your encoder returns already pooled embeddings, this keeps it unchanged.
        if output.ndim == 3:
            output = output[:, 0, :]

        embeddings.append(output.float().cpu())
        all_coords.append(batch_coords.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    coords = torch.cat(all_coords, dim=0).numpy()

    return embeddings, coords


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str,
    random_state: int,
) -> np.ndarray:
    x = StandardScaler().fit_transform(embeddings)

    n_pca = min(50, x.shape[1], x.shape[0])
    x_pca = PCA(n_components=n_pca, random_state=random_state).fit_transform(x)

    if method == "pca":
        return x_pca[:, :2]

    if method == "umap":
        import umap

        return umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(x_pca)

    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=random_state,
            verbose=1,
        ).fit_transform(x_pca)

    raise ValueError(f"Unknown method: {method}")


def save_thumbnails(
    wsi: openslide.OpenSlide,
    coords: np.ndarray,
    out_dir: Path,
    tile_size: int,
    tile_level: int,
    thumbnail_size: int,
) -> list[str]:
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    thumb_paths = []

    for i, (x0, y0) in enumerate(tqdm(coords, desc="Saving tile thumbnails")):
        tile = wsi.read_region(
            (int(x0), int(y0)),
            tile_level,
            (tile_size, tile_size),
        ).convert("RGB")

        tile.thumbnail((thumbnail_size, thumbnail_size))

        filename = f"tile_{i:06d}.jpg"
        path = thumb_dir / filename
        tile.save(path, quality=85)

        thumb_paths.append(f"thumbnails/{filename}")

    return thumb_paths


def save_slide_thumbnail(
    wsi: openslide.OpenSlide,
    out_dir: Path,
    max_size: int = 768,
) -> tuple[str, int, int]:
    """Save a WSI overview thumbnail and return relative path + dimensions."""
    thumb = wsi.get_thumbnail((max_size, max_size)).convert("RGB")

    filename = "slide_thumbnail.jpg"
    path = out_dir / filename
    thumb.save(path, quality=90)

    thumb_w, thumb_h = thumb.size
    return filename, thumb_w, thumb_h


def write_embedding_html(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    slide_thumbnail_path: str,
    slide_width: int,
    slide_height: int,
    slide_thumb_width: int,
    slide_thumb_height: int,
    tile_size_level0: int,
):
    fig = px.scatter(
        df,
        x="plot_x",
        y="plot_y",
        custom_data=["plot_index", "slide_x", "slide_y", "thumbnail"],
        hover_data={
            "plot_x": False,
            "plot_y": False,
            "plot_index": True,
            "slide_x": True,
            "slide_y": True,
            "thumbnail": False,
        },
        title=title,
        width=900,
        height=700,
    )

    fig.update_traces(
        marker=dict(size=5, opacity=0.75),
        hovertemplate=(
            "tile index: %{customdata[0]}<br>"
            "slide_x: %{customdata[1]}<br>"
            "slide_y: %{customdata[2]}<extra></extra>"
        ),
    )

    # Extra trace used to highlight one point when hovering the WSI thumbnail.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=16,
                symbol="circle-open",
                line=dict(width=3),
            ),
            name="Hovered tile",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    plot_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="embedding_plot",
    )

    points = df[
        ["plot_index", "plot_x", "plot_y", "slide_x", "slide_y", "thumbnail"]
    ].to_dict(orient="records")
    points_json = json.dumps(points)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: sans-serif;
    margin: 0;
    display: flex;
    height: 100vh;
    overflow: hidden;
}}

#plot-container {{
    flex: 1;
    min-width: 300px;
    overflow: hidden;
}}

#resizer {{
    width: 6px;
    cursor: col-resize;
    background: #ddd;
    border-left: 1px solid #ccc;
    border-right: 1px solid #ccc;
    flex-shrink: 0;
}}

#resizer:hover {{
    background: #bbb;
}}

#side-panel {{
    width: 560px;
    min-width: 320px;
    max-width: 85vw;
    border-left: 1px solid #ddd;
    padding: 16px;
    box-sizing: border-box;
    background: #fafafa;
    overflow-y: auto;
    flex-shrink: 0;
}}

#tile-preview {{
    width: 256px;
    height: 256px;
    object-fit: contain;
    border: 1px solid #ccc;
    background: white;
}}

#slide-thumb-container {{
    position: relative;
    width: {slide_thumb_width}px;
    height: {slide_thumb_height}px;
    border: 1px solid #ccc;
    background: white;
    margin-top: 8px;
}}

#slide-thumb {{
    position: absolute;
    left: 0;
    top: 0;
    width: {slide_thumb_width}px;
    height: {slide_thumb_height}px;
}}

#slide-overlay {{
    position: absolute;
    left: 0;
    top: 0;
    width: {slide_thumb_width}px;
    height: {slide_thumb_height}px;
    cursor: crosshair;
}}

.meta {{
    margin-top: 12px;
    font-size: 14px;
    line-height: 1.5;
}}

.hint {{
    font-size: 12px;
    color: #666;
    margin-top: 4px;
    margin-bottom: 8px;
}}
</style>
</head>

<body>
<div id="plot-container">
{plot_html}
</div>

<div id="resizer"></div>

<div id="side-panel">
    <h3>Tile preview</h3>
    <img id="tile-preview" src="" alt="hover a point">

    <div class="meta">
        <div><b>Index:</b> <span id="tile-index">-</span></div>
        <div><b>Slide x:</b> <span id="slide-x">-</span></div>
        <div><b>Slide y:</b> <span id="slide-y">-</span></div>
    </div>

    <h3>Slide overview</h3>
    <div class="hint">
        Hover embedding points to locate tiles on the slide.<br>
        Hover the slide thumbnail to highlight the nearest plotted tile.
    </div>

    <div id="slide-thumb-container">
        <img id="slide-thumb" src="{slide_thumbnail_path}" alt="slide thumbnail">
        <canvas
            id="slide-overlay"
            width="{slide_thumb_width}"
            height="{slide_thumb_height}">
        </canvas>
    </div>
</div>

<script>
const plot = document.getElementById("embedding_plot");
const overlay = document.getElementById("slide-overlay");
const ctx = overlay.getContext("2d");

const points = {points_json};

const slideWidth = {slide_width};
const slideHeight = {slide_height};
const thumbWidth = {slide_thumb_width};
const thumbHeight = {slide_thumb_height};
const tileSizeLevel0 = {tile_size_level0};

function slideToThumbX(slideX) {{
    return slideX * thumbWidth / slideWidth;
}}

function slideToThumbY(slideY) {{
    return slideY * thumbHeight / slideHeight;
}}

function tileThumbWidth() {{
    return Math.max(tileSizeLevel0 * thumbWidth / slideWidth, 2);
}}

function tileThumbHeight() {{
    return Math.max(tileSizeLevel0 * thumbHeight / slideHeight, 2);
}}

function drawTileBox(point) {{
    ctx.clearRect(0, 0, thumbWidth, thumbHeight);

    const x = slideToThumbX(Number(point.slide_x));
    const y = slideToThumbY(Number(point.slide_y));
    const w = tileThumbWidth();
    const h = tileThumbHeight();

    ctx.fillStyle = "rgba(255, 0, 0, 0.18)";
    ctx.fillRect(x, y, w, h);

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.beginPath();
    ctx.arc(x + w / 2, y + h / 2, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "blue";
    ctx.fill();
}}

function updateTilePanel(point) {{
    document.getElementById("tile-preview").src = point.thumbnail;
    document.getElementById("tile-index").textContent = point.plot_index;
    document.getElementById("slide-x").textContent = point.slide_x;
    document.getElementById("slide-y").textContent = point.slide_y;
}}

function highlightEmbeddingPoint(point) {{
    // Trace 0 = main scatter
    // Trace 1 = highlight point
    Plotly.restyle(
        plot,
        {{
            x: [[point.plot_x]],
            y: [[point.plot_y]]
        }},
        [1]
    );
}}

function updateFromPoint(point, highlightEmbedding) {{
    updateTilePanel(point);
    drawTileBox(point);

    if (highlightEmbedding) {{
        highlightEmbeddingPoint(point);
    }}
}}

function pointFromCustomData(cd) {{
    const tileIndex = Number(cd[0]);
    return points[tileIndex];
}}

// Direction 1:
// Hover embedding point -> update tile preview + draw rectangle on WSI thumbnail.
plot.on("plotly_hover", function(data) {{
    const point = data.points[0];

    // Ignore hover events from the highlight trace.
    if (point.curveNumber !== 0) {{
        return;
    }}

    const p = pointFromCustomData(point.customdata);
    updateFromPoint(p, false);
}});

function findNearestPointOnSlideThumb(mouseX, mouseY) {{
    const w = tileThumbWidth();
    const h = tileThumbHeight();

    let bestPoint = null;
    let bestDist2 = Infinity;
    let bestInside = false;

    for (const p of points) {{
        const x = slideToThumbX(Number(p.slide_x));
        const y = slideToThumbY(Number(p.slide_y));
        const cx = x + w / 2;
        const cy = y + h / 2;

        const inside =
            mouseX >= x &&
            mouseX <= x + w &&
            mouseY >= y &&
            mouseY <= y + h;

        const dx = mouseX - cx;
        const dy = mouseY - cy;
        const dist2 = dx * dx + dy * dy;

        if (inside && !bestInside) {{
            bestInside = true;
            bestPoint = p;
            bestDist2 = dist2;
            continue;
        }}

        if (inside && bestInside && dist2 < bestDist2) {{
            bestPoint = p;
            bestDist2 = dist2;
            continue;
        }}

        if (!bestInside && dist2 < bestDist2) {{
            bestPoint = p;
            bestDist2 = dist2;
        }}
    }}

    // Avoid selecting a faraway point when the cursor is in empty background.
    // Increase this if the plotted tiles are sparse on the thumbnail.
    const maxDistPx = 20;

    if (!bestInside && bestDist2 > maxDistPx * maxDistPx) {{
        return null;
    }}

    return bestPoint;
}}

// Direction 2:
// Hover slide thumbnail -> find nearest/plotted tile -> highlight embedding point.
overlay.addEventListener("mousemove", function(event) {{
    const rect = overlay.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const p = findNearestPointOnSlideThumb(mouseX, mouseY);

    if (p === null) {{
        return;
    }}

    updateFromPoint(p, true);
}});

// Resizable right panel.
const resizer = document.getElementById("resizer");
const sidePanel = document.getElementById("side-panel");

let isResizing = false;

resizer.addEventListener("mousedown", function(event) {{
    isResizing = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
}});

document.addEventListener("mousemove", function(event) {{
    if (!isResizing) {{
        return;
    }}

    const newWidth = window.innerWidth - event.clientX;
    const minWidth = 320;
    const maxWidth = Math.floor(window.innerWidth * 0.85);

    const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
    sidePanel.style.width = clampedWidth + "px";

    Plotly.Plots.resize(plot);
}});

document.addEventListener("mouseup", function() {{
    if (!isResizing) {{
        return;
    }}

    isResizing = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";

    Plotly.Plots.resize(plot);
}});
</script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")


@click.command()
@click.argument(
    "wsi_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("embedding_explorer"),
    show_default=True,
)
@click.option("--encoder", default="h0-mini", show_default=True)
@click.option("--tile-level", default=0, show_default=True, type=int)
@click.option("--tile-size", default=224, show_default=True, type=int)
@click.option("--step-size", default=224, show_default=True, type=int)
@click.option("--max-points", default=5000, show_default=True, type=int)
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--num-workers", default=4, show_default=True, type=int)
@click.option(
    "--method",
    type=click.Choice(["pca", "umap", "tsne"]),
    default="umap",
    show_default=True,
)
@click.option("--device", default="cuda", show_default=True)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--slide-thumb-size", default=768, show_default=True, type=int)
def main(
    wsi_path: Path,
    out_dir: Path,
    encoder: str,
    tile_level: int,
    tile_size: int,
    step_size: int,
    max_points: int,
    batch_size: int,
    num_workers: int,
    method: str,
    device: str,
    seed: int,
    slide_thumb_size: int,
):
    slide_name = wsi_path.stem
    slide_out_dir = out_dir / slide_name
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Opening WSI: {wsi_path}")
    wsi = openslide.OpenSlide(str(wsi_path))

    click.echo("Generating tile coordinates...")
    coords = tile_slide(
        wsi=wsi,
        tile_level=tile_level,
        tile_size=tile_size,
        step_size=step_size,
    )

    click.echo(f"Found {len(coords)} tissue tiles")

    rng = np.random.default_rng(seed)

    if len(coords) > max_points:
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords = coords[idx]

    click.echo(f"Using {len(coords)} plotted tiles")

    click.echo(f"Loading encoder: {encoder}")
    model, model_attrs = build_encoder(encoder)
    model.to(device)
    click.echo(
        f"Model loaded (Num Parameters: {sum(p.numel() for p in model.parameters())}) with pixel_mean={model_attrs['pixel_mean']} and pixel_std={model_attrs['pixel_std']}"
    )

    transform = build_tile_transform(
        px_mean=model_attrs["pixel_mean"],
        px_std=model_attrs["pixel_std"],
    )

    dataset = TileDataset(
        wsi=wsi,
        coords=coords,
        tile_size=tile_size,
        tile_level=tile_level,
        transforms=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    embeddings, coords_used = compute_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    np.save(slide_out_dir / "embeddings.npy", embeddings)
    np.save(slide_out_dir / "coords.npy", coords_used)

    click.echo(f"Reducing embeddings with {method}...")
    plot_coords = reduce_embeddings(
        embeddings=embeddings,
        method=method,
        random_state=seed,
    )

    thumb_paths = save_thumbnails(
        wsi=wsi,
        coords=coords_used,
        out_dir=slide_out_dir,
        tile_size=tile_size,
        tile_level=tile_level,
        thumbnail_size=256,
    )

    slide_thumbnail_path, slide_thumb_w, slide_thumb_h = save_slide_thumbnail(
        wsi=wsi,
        out_dir=slide_out_dir,
        max_size=slide_thumb_size,
    )

    slide_w, slide_h = wsi.dimensions
    tile_size_level0 = int(round(tile_size * float(wsi.level_downsamples[tile_level])))

    df = pd.DataFrame(
        {
            "plot_index": np.arange(len(coords_used)),
            "plot_x": plot_coords[:, 0],
            "plot_y": plot_coords[:, 1],
            "slide_x": coords_used[:, 0],
            "slide_y": coords_used[:, 1],
            "thumbnail": thumb_paths,
        }
    )

    df.to_csv(slide_out_dir / "points.csv", index=False)

    config = {
        "wsi_path": str(wsi_path),
        "encoder": encoder,
        "tile_level": tile_level,
        "tile_size": tile_size,
        "step_size": step_size,
        "tile_size_level0": tile_size_level0,
        "max_points": max_points,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "method": method,
        "device": device,
        "seed": seed,
        "slide_width": slide_w,
        "slide_height": slide_h,
        "slide_thumb_width": slide_thumb_w,
        "slide_thumb_height": slide_thumb_h,
        "slide_thumb_size": slide_thumb_size,
    }

    (slide_out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    html_path = slide_out_dir / f"{method}_with_tiles.html"

    write_embedding_html(
        df=df,
        out_path=html_path,
        title=f"{method.upper()} embedding explorer - {slide_name}",
        slide_thumbnail_path=slide_thumbnail_path,
        slide_width=slide_w,
        slide_height=slide_h,
        slide_thumb_width=slide_thumb_w,
        slide_thumb_height=slide_thumb_h,
        tile_size_level0=tile_size_level0,
    )

    click.echo(f"Done: {html_path.resolve()}")


if __name__ == "__main__":
    main()
