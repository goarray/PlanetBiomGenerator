import json
import sys
import subprocess
from pathlib import Path
from PlanetNewsfeed import handle_news
from PlanetConstants import (
    get_config,
    SCRIPT_DIR,
    CONFIG_PATH,
    PLUGINS_DIR,
    BIOM_DIR,
    MATERIAL_PATH,
)

# Core directories
bundle_dir = getattr(sys, "_MEIPASS", None)
if bundle_dir:
    BASE_DIR = Path(bundle_dir).resolve()
else:
    BASE_DIR = Path(__file__).resolve().parent


config = get_config()


def replace_placeholders_recursive(data, plugin_name: str, planet_name: str) -> None:
    """Recursively replace plugin_name and planet_name placeholders in all strings."""
    handle_news(None)
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                if "plugin_name" in value or "planet_name" in value:
                    new_value = value.replace("plugin_name", plugin_name).replace(
                        "planet_name", planet_name
                    )
                    data[key] = new_value
                    handle_news(
                        None,
                        "info",
                        f"Replaced in {key}: '{value}' -> '{new_value}'",
                    )
            else:
                replace_placeholders_recursive(value, plugin_name, planet_name)
    elif isinstance(data, list):
        for item in data:
            replace_placeholders_recursive(item, plugin_name, planet_name)


def count_json_lines(data) -> int:
    """Count the number of lines in a JSON object when serialized."""
    return len(json.dumps(data, indent=4).splitlines())


def get_planet_material_paths():
    """Yield (planet_name, texture_path, material_path)."""
    plugin_name = config.get("plugin_name", "PLUGINNOTFOUND")
    planet_name = config.get("planet_name", "PLUGINNOTFOUND")

    texture_path = (
        PLUGINS_DIR
        / plugin_name
        / "textures"
        / plugin_name
        / planet_name
        / f"{planet_name}_color.dds"
    )

    material_path = (
        PLUGINS_DIR
        / plugin_name
        / "materials"
        / plugin_name
        / "planets"
        / f"{planet_name}.mat"
    )

    handle_news(None, "info", f"Material Path: {material_path}")

    write_material_file(material_path, plugin_name, planet_name)

    yield planet_name, texture_path, material_path


def write_material_file(material_path, plugin_name, planet_name):
    """Generate and write a formatted .mat file with replaced placeholders."""
    output_path = Path(material_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = MATERIAL_PATH
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            mat_data = json.load(f)

        initial_lines = count_json_lines(mat_data)
        handle_news(None, "info", f"Template JSON line count: {initial_lines}")

        handle_news(
            None,
            "info",
            f"Replacing placeholders for plugin_name={plugin_name}, planet_name={planet_name}",
        )
        replace_placeholders_recursive(mat_data, plugin_name, planet_name)

        replaced_lines = count_json_lines(mat_data)
        handle_news(None, "info", f"Post-replacement JSON line count: {replaced_lines}")

        mat_data["Filename"] = f"MATERIALS\\{plugin_name}\\planets\\{planet_name}.mat"

        for obj in mat_data.get("Objects", []):
            for component in obj.get("Components", []):
                if component.get("Type") == "BSMaterial::MRTextureFile":
                    file_data = component.get("Data", {})
                    if "FileName" in file_data:
                        file_data["FileName"] = (
                            file_data["FileName"]
                            .replace("plugin_name", plugin_name)
                            .replace("planet_name", planet_name)
                        )

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(mat_data, f, indent=4, ensure_ascii=False)

        handle_news(None, "info", f"Material file written to: {output_path}")

    except FileNotFoundError:
        print(f"Template file not found: {template_path}", file=sys.stderr)


def main():
    """Run material processing and print results."""
    print("=== Starting Material Path Processing ===")
    for planet_name, texture_path, material_path in get_planet_material_paths():
        handle_news(
            None, "info", f"Updated {planet_name} material. Saved at: {material_path}"
        )

    if config.get("run_planet_meshes", True):
        subprocess.run([sys.executable, str(SCRIPT_DIR / "PlanetMeshes.py")], check=True)
    else:
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
