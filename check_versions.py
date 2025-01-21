import tomli
from importlib.metadata import version
from packaging.requirements import Requirement

def get_installed_version(package_name):
    """Get installed version of a package, return None if not installed."""
    try:
        return version(package_name.lower())
    except Exception:
        return None

def parse_requirement(req_string):
    """Parse requirement string to get package name."""
    # Remove any markers or extras
    req_string = req_string.split(';')[0].strip()
    return Requirement(req_string).name

if __name__ == "__main__":
    # Read pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    # Get dependencies list
    dependencies = pyproject["project"]["dependencies"]

    # Print header
    print("\nDependency Versions Check")
    print("-" * 60)
    print(f"{'Package':<30} {'Required':<20} {'Installed':<15}")
    print("-" * 60)

    # Check each dependency
    for dep in dependencies:
        try:
            package_name = parse_requirement(dep)
            installed_version = get_installed_version(package_name)
            required = dep.replace(package_name, "").strip()
            print(f"{package_name:<30} {required:<20} {installed_version or 'Not installed':<15}")
        except Exception as e:
            print(f"Error parsing {dep}: {str(e)}")

    print("-" * 60)
