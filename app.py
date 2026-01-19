import os

# Must be set before importing matplotlib via create_map_poster.py
os.environ.setdefault("MPLBACKEND", "Agg")

import re
import tempfile
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path

import gradio as gr
from geopy.geocoders import Nominatim
from pydantic import BaseModel, ValidationError, field_validator

import osmnx as ox

import create_map_poster as maptoposter


APP_TITLE = "MapToPoster"
DEFAULT_DISTANCE_M = 10000
MIN_DISTANCE_M = 2000
MAX_DISTANCE_M = 20000
DEFAULT_DPI = 300
MIN_DPI = 150
MAX_DPI = 600
DEFAULT_DOT_SIZE = 60
MIN_DOT_SIZE = 10
MAX_DOT_SIZE = 300


class NetworkType(str, Enum):
    ALL = "all"
    ALL_PUBLIC = "all_public"
    DRIVE = "drive"
    DRIVE_SERVICE = "drive_service"
    WALK = "walk"
    BIKE = "bike"


class DistanceType(str, Enum):
    BBOX = "bbox"
    NETWORK = "network"


class Orientation(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


NETWORK_TYPES = [item.value for item in NetworkType]
DIST_TYPES = [item.value for item in DistanceType]
ORIENTATION_TYPES = [item.value for item in Orientation]


_REPO_ROOT = Path(__file__).resolve().parent


class GenerateRequest(BaseModel):
    city: str
    country: str
    theme: str
    distance_m: int
    dpi: int
    network_type: NetworkType
    dist_type: DistanceType
    orientation: Orientation
    home_point: str | None = None
    dot_size: float | None = None

    @field_validator("city", "country", "theme")
    @classmethod
    def _strip_and_require(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("must be provided")
        return value

    @field_validator("distance_m")
    @classmethod
    def _validate_distance(cls, value: int) -> int:
        value = int(value)
        if value < MIN_DISTANCE_M or value > MAX_DISTANCE_M:
            raise ValueError(
                f"Distance must be between {MIN_DISTANCE_M} and {MAX_DISTANCE_M} meters."
            )
        return value

    @field_validator("dpi")
    @classmethod
    def _validate_dpi(cls, value: int) -> int:
        value = int(value)
        if value < MIN_DPI or value > MAX_DPI:
            raise ValueError(f"DPI must be between {MIN_DPI} and {MAX_DPI}.")
        return value

    @field_validator("network_type", mode="before")
    @classmethod
    def _validate_network_type(cls, value: str | NetworkType) -> NetworkType:
        if isinstance(value, NetworkType):
            return value
        value = (value or "").strip()
        try:
            return NetworkType(value)
        except ValueError as exc:
            raise ValueError("Invalid network type.") from exc

    @field_validator("dist_type", mode="before")
    @classmethod
    def _validate_dist_type(cls, value: str | DistanceType) -> DistanceType:
        if isinstance(value, DistanceType):
            return value
        value = (value or "").strip()
        try:
            return DistanceType(value)
        except ValueError as exc:
            raise ValueError("Invalid distance type.") from exc

    @field_validator("orientation", mode="before")
    @classmethod
    def _validate_orientation(cls, value: str | Orientation) -> Orientation:
        if isinstance(value, Orientation):
            return value
        value = (value or "").strip()
        try:
            return Orientation(value)
        except ValueError as exc:
            raise ValueError("Invalid orientation.") from exc

    @field_validator("dot_size", mode="before")
    @classmethod
    def _validate_dot_size(cls, value: str | float | None) -> float | None:
        if value is None or value == "":
            return None
        value = float(value)
        if value < MIN_DOT_SIZE or value > MAX_DOT_SIZE:
            raise ValueError(f"Dot size must be between {MIN_DOT_SIZE} and {MAX_DOT_SIZE}.")
        return value

def _parse_home_point(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 2:
        raise ValueError("Home point must be in 'lat, lon' format.")
    lat = float(parts[0])
    lon = float(parts[1])
    if lat < -90 or lat > 90:
        raise ValueError("Home point latitude must be between -90 and 90.")
    if lon < -180 or lon > 180:
        raise ValueError("Home point longitude must be between -180 and 180.")
    return (lat, lon)


def _load_readme_example_posters() -> list[tuple[str, str]]:
    """Return (absolute_path, caption) pairs for example posters referenced in README."""

    readme_path = _REPO_ROOT / "README.md"
    if not readme_path.exists():
        return []

    try:
        content = readme_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    # Keep order stable and match the README's poster examples.
    matches = re.findall(r"(?i)\bposters/[^\"'\s>\)]+\.png\b", content)
    seen: set[str] = set()
    posters: list[tuple[str, str]] = []
    for rel in matches:
        if rel in seen:
            continue
        seen.add(rel)

        abs_path = (_REPO_ROOT / rel).resolve()
        if not abs_path.exists():
            continue

        caption = Path(rel).stem.replace("_", " ")
        posters.append((str(abs_path), caption))

    return posters


def _load_local_example_posters(limit: int = 18) -> list[tuple[str, str]]:
    """Return (absolute_path, caption) pairs for local posters/ examples."""

    posters_dir = _REPO_ROOT / "posters"
    if not posters_dir.exists():
        return []

    paths = sorted(
        posters_dir.glob("*.png"),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0),
        reverse=True,
    )
    posters: list[tuple[str, str]] = []
    for path in paths[: max(0, int(limit))]:
        caption = path.stem.replace("_", " ")
        posters.append((str(path.resolve()), caption))
    return posters


def _configure_osmnx_cache() -> None:
    cache_dir = os.environ.get("OSMNX_CACHE_DIR", "/tmp/osmnx_cache")
    os.makedirs(cache_dir, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    ox.settings.log_console = False


_configure_osmnx_cache()


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_\-]", "", value)
    return value or "map"


_last_geocode_ts = 0.0


def _rate_limit_geocode(min_interval_s: float = 1.05) -> None:
    global _last_geocode_ts
    now = time.time()
    wait_s = (_last_geocode_ts + min_interval_s) - now
    if wait_s > 0:
        time.sleep(wait_s)
    _last_geocode_ts = time.time()


@lru_cache(maxsize=1)
def _geocoder() -> Nominatim:
    user_agent = os.environ.get("MAPTOP_POSTER_USER_AGENT", "maptoposter-hf-space")
    return Nominatim(user_agent=user_agent)


@lru_cache(maxsize=256)
def _geocode(city: str, country: str) -> maptoposter.Coordinates:
    _rate_limit_geocode()
    location = _geocoder().geocode(f"{city}, {country}")
    if not location:
        raise ValueError(f"Could not find coordinates for {city}, {country}")
    return maptoposter.Coordinates(
        lat=float(location.latitude),
        lon=float(location.longitude),
    )


def generate(
    city: str,
    country: str,
    theme: str,
    distance_m: int,
    dpi: int,
    network_type: str,
    dist_type: str,
    orientation: str,
    home_point: str | None,
    dot_size: float | None,
) -> str:
    try:
        request = GenerateRequest(
            city=city,
            country=country,
            theme=theme,
            distance_m=distance_m,
            dpi=dpi,
            network_type=network_type,
            dist_type=dist_type,
            orientation=orientation,
            home_point=home_point,
            dot_size=dot_size,
        )
    except ValidationError as exc:
        raise gr.Error(str(exc))

    try:
        dot_coords = _parse_home_point(request.home_point)
    except ValueError as exc:
        raise gr.Error(str(exc))

    available_themes = maptoposter.get_available_themes()
    if request.theme not in available_themes:
        raise gr.Error(f"Unknown theme: {request.theme}")

    maptoposter.THEME = maptoposter.load_theme(request.theme)

    coords = _geocode(request.city, request.country)

    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(
        tmp_dir,
        f"{_slugify(request.city)}_{_slugify(request.theme)}_{int(time.time())}.png",
    )

    maptoposter.create_poster(
        request.city,
        request.country,
        coords,
        request.distance_m,
        output_path,
        network_type=request.network_type.value,
        dist_type=request.dist_type.value,
        dpi=request.dpi,
        dot=dot_coords,
        dot_size=request.dot_size if request.dot_size is not None else DEFAULT_DOT_SIZE,
        orientation=request.orientation.value,
    )
    return output_path


def build_demo() -> gr.Blocks:
    themes = maptoposter.get_available_themes()
    if not themes:
        themes = ["feature_based"]
    default_theme = "feature_based" if "feature_based" in themes else themes[0]

    readme_examples = _load_readme_example_posters()
    local_examples = _load_local_example_posters(limit=18)
    example_posters = readme_examples or local_examples

    css = """
    :root {
      --bg0: #070A12;
      --bg1: #0B1020;
      --card: rgba(255,255,255,0.06);
      --border: rgba(255,255,255,0.10);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.70);
      --accent: #7C5CFF;
      --accent2: #22D3EE;
    }

    body {
      background:
        radial-gradient(900px 600px at 10% 0%, rgba(124,92,255,0.20) 0%, rgba(124,92,255,0) 60%),
        radial-gradient(900px 600px at 90% 10%, rgba(34,211,238,0.16) 0%, rgba(34,211,238,0) 55%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%) !important;
    }

    .gradio-container {
      max-width: 1120px !important;
      margin: 0 auto !important;
      color: var(--text);
    }

    .mtp-hero {
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      border-radius: 18px;
      padding: 22px 22px 18px;
      margin-bottom: 14px;
      box-shadow: 0 12px 35px rgba(0,0,0,0.35);
    }

    .mtp-hero h1 {
      font-size: 34px;
      line-height: 1.1;
      margin: 0;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #FFFFFF 0%, rgba(255,255,255,0.86) 40%, rgba(34,211,238,0.88) 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .mtp-hero p {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }

    .mtp-badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    .mtp-badge {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: rgba(255,255,255,0.80);
      font-size: 12px;
    }

    .mtp-card {
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.30);
    }

    .mtp-primary button {
      background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
      border: none !important;
      color: #061018 !important;
      font-weight: 700 !important;
      border-radius: 14px !important;
      padding: 12px 14px !important;
    }

    .mtp-primary button:hover { filter: brightness(1.05); }

    .mtp-subtle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
    }

    label span {
      font-weight: 600 !important;
      letter-spacing: -0.01em;
    }

    .mtp-gallery .grid { gap: 10px !important; }
    """

    theme_obj = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size="lg",
        text_size="md",
    )

    with gr.Blocks(title=APP_TITLE, theme=theme_obj, css=css) as demo:
        gr.HTML(
            """
            <div class="mtp-hero">
              <h1>MapToPoster</h1>
              <p>Generate minimalist city map posters from OpenStreetMap data — fast, clean, and print-ready.</p>
              <div class="mtp-badges">
                <span class="mtp-badge">Queued generation</span>
                <span class="mtp-badge">Cached OSM requests</span>
                <span class="mtp-badge">Best with 2–20 km radius</span>
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["mtp-card"]):
                    with gr.Row():
                        city = gr.Textbox(label="City", placeholder="Barcelona", value="Barcelona")
                        country = gr.Textbox(label="Country", placeholder="Spain", value="Spain")

                    theme = gr.Dropdown(label="Theme", choices=themes, value=default_theme)
                    distance = gr.Slider(
                        label="Radius (meters)",
                        minimum=MIN_DISTANCE_M,
                        maximum=MAX_DISTANCE_M,
                        step=500,
                        value=DEFAULT_DISTANCE_M,
                    )

                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Row():
                            dpi = gr.Slider(
                                label="DPI",
                                minimum=MIN_DPI,
                                maximum=MAX_DPI,
                                step=10,
                                value=DEFAULT_DPI,
                            )
                            network_type = gr.Dropdown(
                                label="Network type",
                                choices=NETWORK_TYPES,
                                value="all",
                            )
                        dist_type = gr.Dropdown(
                            label="Distance type",
                            choices=DIST_TYPES,
                            value="bbox",
                        )
                        orientation = gr.Dropdown(
                            label="Orientation",
                            choices=ORIENTATION_TYPES,
                            value="portrait",
                        )
                        home_point = gr.Textbox(
                            label="Home point (lat, lon)",
                            placeholder="31.3, 2.3",
                        )
                        dot_size = gr.Slider(
                            label="Dot size",
                            minimum=MIN_DOT_SIZE,
                            maximum=MAX_DOT_SIZE,
                            step=5,
                            value=DEFAULT_DOT_SIZE,
                        )

                    btn = gr.Button("Generate poster", elem_classes=["mtp-primary"])
                    gr.HTML(
                        """<div class="mtp-subtle">Uses public geocoding + OSM services. Please keep distances modest.</div>"""
                    )

            with gr.Column(scale=6):
                with gr.Group(elem_classes=["mtp-card"]):
                    out = gr.Image(label="Poster", type="filepath", show_label=True)

        btn.click(
            generate,
            inputs=[
                city,
                country,
                theme,
                distance,
                dpi,
                network_type,
                dist_type,
                orientation,
                home_point,
                dot_size,
            ],
            outputs=[out],
        )

        if example_posters:
            gr.Markdown("## Example gallery")
            with gr.Group(elem_classes=["mtp-card", "mtp-gallery"]):
                gr.Gallery(
                    value=example_posters,
                    columns=4,
                    height=320,
                    show_label=False,
                )

        demo.queue(max_size=16, default_concurrency_limit=1)

    return demo


demo = build_demo()


if __name__ == "__main__":
    def _default_server_name() -> str:
        # Hugging Face Spaces expects binding on 0.0.0.0.
        if os.environ.get("SPACE_ID"):
            return "0.0.0.0"
        return "127.0.0.1"

    server_name = os.environ.get("HOST") or _default_server_name()
    demo.launch(server_name=server_name, server_port=int(os.environ.get("PORT", "7860")))
