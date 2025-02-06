from pathlib import Path

def setup_project_directories(base_dir=None):
    """
    Creates a structured project directory with default folders. Ensures existing directories 
    with data are not overwritten. Returns a dictionary of created directories.

    Parameters:
    - base_dir (str or Path, optional): Path to the base directory. If None, asks user for input.

    Returns:
    - dict: Dictionary containing the paths of created directories.
    """

    # Assign or ask for base directory
    if base_dir is None:
        base_dir = input(f"Enter the base directory (default: {Path.cwd()}): ").strip()
        base_dir = Path(base_dir) if base_dir else Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Validate path
    if not base_dir.exists():
        print(f" Warning: The specified base directory does not exist. Creating it at: {base_dir}")
        base_dir.mkdir(parents=True, exist_ok=True)

    # Define useful subdirectories
    subdirs = {
        "Data": base_dir / "data",
        "Raw Data": base_dir / "data" / "raw",
        "Processed Data": base_dir / "data" / "processed",
        "Output": base_dir / "output",
        "Figures": base_dir / "output" / "figures",
        "Logs": base_dir / "logs",
        "Config": base_dir / "config",
        "Scripts": base_dir / "scripts",
        "Notebooks": base_dir / "notebooks",
        "Reports": base_dir / "reports",
    }

    # Track created directories
    created_dirs = {}

    # Create directories safely
    for name, path in subdirs.items():
        if path.exists() and any(path.iterdir()):  # Check if directory exists and has files
            print(f"Warning: '{name}' directory already exists and contains data. Skipping creation.")
        else:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs[name] = path  # Store successfully created directories

    # Print Summary
    print(f"Base Directory: {base_dir.resolve()}\n")

    for name, path in subdirs.items():
        status = "✅ Created" if name in created_dirs else "⚠️ Exists (Not Modified)"
        print(f"   📁 {name}: {path.relative_to(base_dir)}  - {status}")

    print("\n Setup complete!")

    '''
    # Example Usage
    project_dirs = setup_project_directories("/Users/yourname/projects")

    # Use returned dictionary for path references
    dem_path = project_dirs["Data"] / "input_dem.tif"
    print(f"\nDEM will be stored in: {dem_path}")
    '''
    return subdirs  # Return the full dictionary of directories
