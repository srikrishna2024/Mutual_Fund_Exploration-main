import os
import ast
import subprocess
import sys

def get_imports_from_file(filepath):
    """Extract all imports from a Python file."""
    imports = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])  # Only take the base package name
                elif isinstance(node, ast.ImportFrom):
                    if node.module:  # Some ImportFrom nodes might have module=None
                        imports.add(node.module.split('.')[0])  # Only take the base package name
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        
    return imports

def scan_folder_for_python_files(folder_path):
    """Recursively scan folder and subfolders for Python scripts."""
    python_files = []
    
    # Walk through the directory, including subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def generate_requirements_txt(folder_path, output_path):
    """Generate a requirements.txt file for all Python dependencies."""
    all_imports = set()
    
    # Scan for Python scripts in the folder and subfolders
    python_files = scan_folder_for_python_files(folder_path)
    if not python_files:
        print(f"No Python files found in {folder_path}. Please check the folder path.")
        return
    
    print(f"Found {len(python_files)} Python files. Scanning for imports...")
    
    # Extract imports from each Python file
    for python_file in python_files:
        imports = get_imports_from_file(python_file)
        if imports:
            print(f"Found imports in {python_file}: {imports}")
        all_imports.update(imports)
    
    # Filter out standard library modules
    stdlib_modules = set(sys.builtin_module_names)
    external_imports = sorted([imp for imp in all_imports if imp not in stdlib_modules])
    
    # Create a requirements.txt file with the list of unique imports
    if external_imports:
        requirements_path = os.path.join(output_path, 'requirements.txt')
        with open(requirements_path, 'w') as req_file:
            for imp in external_imports:
                req_file.write(f'{imp}\n')
        print(f"\nrequirements.txt has been generated at {requirements_path} with the following dependencies:")
        for imp in external_imports:
            print(f"- {imp}")
    else:
        print("\nNo external imports were found in any Python files.")

def install_requirements(output_path):
    """Install all packages listed in requirements.txt."""
    requirements_path = os.path.join(output_path, 'requirements.txt')
    if not os.path.exists(requirements_path):
        print(f"\nError: {requirements_path} not found. Please generate it first.")
        return
    
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("All dependencies have been successfully installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

def main():
    # Get the folder where the script is located for output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ask user for source folder path
    print("Python Dependency Scanner")
    print("------------------------")
    folder_path = input("Enter the path to scan for Python files: ").strip()
    
    # Validate the folder path
    if not os.path.isdir(folder_path):
        print(f"\nError: The path '{folder_path}' is not a valid directory.")
        return
    
    # Ask the user for their choice
    print("\nChoose an option:")
    print("1. Generate requirements.txt file")
    print("2. Generate requirements.txt file and install all packages")
    
    user_choice = input("Enter the number (1 or 2): ").strip()

    if user_choice == '1':
        generate_requirements_txt(folder_path, script_dir)
    elif user_choice == '2':
        generate_requirements_txt(folder_path, script_dir)
        install_requirements(script_dir)
    else:
        print("Invalid option. Please choose either 1 or 2.")

if __name__ == "__main__":
    main()