import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from dataclasses import dataclass
from datetime import datetime
import argparse
from functools import lru_cache
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"


def _configure_osmnx_cache() -> None:
    cache_dir = os.environ.get("OSMNX_CACHE_DIR", "/tmp/osmnx_cache")
    os.makedirs(cache_dir, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    ox.settings.log_console = False


_configure_osmnx_cache()


@dataclass(frozen=True)
class FontPaths:
    bold: str
    regular: str
    light: str


@dataclass(frozen=True)
class Coordinates:
    lat: float
    lon: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.lat, self.lon)


@dataclass(frozen=True)
class PosterRequest:
    city: str
    country: str
    theme: str
    distance_m: int


class ThemeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "Feature-Based Shading"
    description: Optional[str] = None
    bg: str = "#FFFFFF"
    text: str = "#000000"
    gradient_color: str = "#FFFFFF"
    water: str = "#C0C0C0"
    parks: str = "#F0F0F0"
    road_motorway: str = "#0A0A0A"
    road_primary: str = "#1A1A1A"
    road_secondary: str = "#2A2A2A"
    road_tertiary: str = "#3A3A3A"
    road_residential: str = "#4A4A4A"
    road_default: str = "#3A3A3A"

def load_fonts() -> Optional[FontPaths]:
    """
    Load Roboto fonts from the fonts directory.
    Returns FontPaths for different weights.
    """
    fonts = FontPaths(
        bold=os.path.join(FONTS_DIR, "Roboto-Bold.ttf"),
        regular=os.path.join(FONTS_DIR, "Roboto-Regular.ttf"),
        light=os.path.join(FONTS_DIR, "Roboto-Light.ttf"),
    )

    for path in (fonts.bold, fonts.regular, fonts.light):
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None

    return fonts

FONTS = load_fonts()


@lru_cache(maxsize=1)
def _geocoder() -> Nominatim:
    user_agent = os.environ.get("MAPTOP_POSTER_USER_AGENT", "maptoposter-cli")
    return Nominatim(user_agent=user_agent)

def generate_output_filename(city: str, theme_name: str) -> str:
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(' ', '_')
    filename = f"{city_slug}_{theme_name}_{timestamp}.png"
    return os.path.join(POSTERS_DIR, filename)

def get_available_themes() -> list[str]:
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes

def load_theme(theme_name: str = "feature_based") -> ThemeConfig:
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return ThemeConfig()
    
    with open(theme_file, "r") as f:
        raw_theme = json.load(f)
        try:
            theme = ThemeConfig(**raw_theme)
        except ValidationError as exc:
            print(f"⚠ Theme file '{theme_file}' is invalid: {exc}")
            theme = ThemeConfig()
        print(f"✓ Loaded theme: {theme.name or theme_name}")
        if theme.description:
            print(f"  {theme.description}")
        return theme

# Load theme (can be changed via command line or input)
THEME: Optional[ThemeConfig] = None  # Will be loaded later


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city \"New York\" --country \"USA\"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """,
    )

    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')

    args = parser.parse_args(argv)

    # If no arguments provided, show examples
    if argv is None and len(os.sys.argv) == 1:
        print_examples()
        return 0

    # List themes if requested
    if args.list_themes:
        list_themes()
        return 0

    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        return 1

    # Validate theme exists
    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        return 1

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    # Load theme
    global THEME
    THEME = load_theme(args.theme)

    # Get coordinates and generate poster
    try:
        request = PosterRequest(
            city=args.city,
            country=args.country,
            theme=args.theme,
            distance_m=args.distance,
        )
        coords = get_coordinates(request.city, request.country)
        output_file = generate_output_filename(request.city, request.theme)
        create_poster(request.city, request.country, coords, request.distance_m, output_file)

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def create_gradient_fade(ax, color: str, location: str = "bottom", zorder: int = 10) -> None:
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], 
              aspect='auto', cmap=custom_cmap, zorder=zorder, origin='lower')

def _require_theme() -> ThemeConfig:
    if THEME is None:
        raise RuntimeError("Theme is not loaded. Call load_theme() first.")
    return THEME


def get_edge_colors_by_type(G) -> list[str]:
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    theme = _require_theme()
    edge_colors: list[str] = []
    
    for u, v, data in G.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')
        
        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign color based on road type
        if highway in ["motorway", "motorway_link"]:
            color = theme.road_motorway
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = theme.road_primary
        elif highway in ["secondary", "secondary_link"]:
            color = theme.road_secondary
        elif highway in ["tertiary", "tertiary_link"]:
            color = theme.road_tertiary
        elif highway in ["residential", "living_street", "unclassified"]:
            color = theme.road_residential
        else:
            color = theme.road_default
        
        edge_colors.append(color)
    
    return edge_colors

def get_edge_widths_by_type(G) -> list[float]:
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths: list[float] = []
    
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        
        # Assign width based on road importance
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        
        edge_widths.append(width)
    
    return edge_widths

@lru_cache(maxsize=256)
def get_coordinates(city: str, country: str) -> Coordinates:
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    print("Looking up coordinates...")
    geolocator = _geocoder()

    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    location = geolocator.geocode(f"{city}, {country}")
    
    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return Coordinates(lat=float(location.latitude), lon=float(location.longitude))
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")

def _coerce_coordinates(point: Coordinates | Sequence[float]) -> Coordinates:
    if isinstance(point, Coordinates):
        return point
    if isinstance(point, Sequence) and len(point) == 2:
        return Coordinates(lat=float(point[0]), lon=float(point[1]))
    raise TypeError("point must be Coordinates or (lat, lon) sequence")


def create_poster(
    city: str,
    country: str,
    point: Coordinates | Sequence[float],
    dist: int,
    output_file: str,
    *,
    network_type: str = "all",
    dist_type: str = "bbox",
    dpi: int = 300,
) -> None:
    print(f"\nGenerating map for {city}, {country}...")
    theme = _require_theme()
    coords = _coerce_coordinates(point)
    
    # Progress bar for data fetching
    with tqdm(total=3, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(
            coords.as_tuple(),
            dist=dist,
            dist_type=dist_type,
            network_type=network_type,
        )
        pbar.update(1)
        time.sleep(0.5)  # Rate limit between requests
        
        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(coords.as_tuple(), tags={"natural": "water", "waterway": "riverbank"}, dist=dist)
        except:
            water = None
        pbar.update(1)
        time.sleep(0.3)
        
        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = ox.features_from_point(coords.as_tuple(), tags={"leisure": "park", "landuse": "grass"}, dist=dist)
        except:
            parks = None
        pbar.update(1)
    
    print("✓ All data downloaded successfully!")
    
    # 2. Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(12, 16), facecolor=theme.bg)
    ax.set_facecolor(theme.bg)
    ax.set_position([0, 0, 1, 1])
    
    # 3. Plot Layers
    # Layer 1: Polygons
    if water is not None and not water.empty:
        water.plot(ax=ax, facecolor=theme.water, edgecolor="none", zorder=1)
    if parks is not None and not parks.empty:
        parks.plot(ax=ax, facecolor=theme.parks, edgecolor="none", zorder=2)
    
    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G)
    
    ox.plot_graph(
        G, ax=ax, bgcolor=theme.bg,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False
    )
    
    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, theme.gradient_color, location="bottom", zorder=10)
    create_gradient_fade(ax, theme.gradient_color, location="top", zorder=10)
    
    # 4. Typography using Roboto font
    if FONTS:
        font_main = FontProperties(fname=FONTS.bold, size=60)
        font_top = FontProperties(fname=FONTS.bold, size=40)
        font_sub = FontProperties(fname=FONTS.light, size=22)
        font_coords = FontProperties(fname=FONTS.regular, size=14)
    else:
        # Fallback to system fonts
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_top = FontProperties(family='monospace', weight='bold', size=40)
        font_sub = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)
    
    spaced_city = "  ".join(list(city.upper()))

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=theme.text, ha="center", fontproperties=font_main, zorder=11)
    
    ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
            color=theme.text, ha="center", fontproperties=font_sub, zorder=11)
    
    lat, lon = coords.lat, coords.lon
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")
    
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=theme.text, alpha=0.7, ha="center", fontproperties=font_coords, zorder=11)
    
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
            color=theme.text, linewidth=1, zorder=11)

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS.light, size=8)
    else:
        font_attr = FontProperties(family='monospace', size=8)
    
    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=theme.text, alpha=0.5, ha="right", va="bottom", 
            fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=int(dpi), facecolor=theme.bg)
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")

def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --theme, -t       Theme name (default: feature_based)
  --distance, -d    Map radius in meters (default: 29000)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r") as f:
                theme_data = json.load(f)
                theme = ThemeConfig(**theme_data)
                display_name = theme.name or theme_name
                description = theme.description or ""
        except Exception:
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    raise SystemExit(main())
