import os
import subprocess
import argparse
import pkg_resources

def run_pipreqs(directory, temp_file):
    """Run pipreqs on the given directory and output to a temporary file."""
    subprocess.run(["pipreqs", directory, "--force", "--savepath", temp_file, "--scan-notebooks"], check=True)

def merge_requirements(temp_files, output_file):
    """Merge requirements from temporary files into the output file, removing duplicates."""
    requirements_set = set()
    
    for temp_file in temp_files:
        with open(temp_file, 'r') as temp:
            requirements = temp.readlines()
            requirements_set.update(requirements)
    
    # Add nbformat version
    try:
        nbformat_version = pkg_resources.get_distribution("nbformat").version
        requirements_set.add(f"nbformat=={nbformat_version}\n")
    except pkg_resources.DistributionNotFound:
        print("nbformat is not installed.")
    
    # Check for starterkits package
    replacers = {'starterkits': "starterkits @ git+https://github.com/EluciDATALab/elucidatalab.starterkits.git",
                 'tqdm': "tqdm==4.66.4",
                 'ipypb': "ipypb==0.5.2"}
    for k, v in replacers.items():
        try:
            pkg_resources.get_distribution(k)
            requirements_set = {req for req in requirements_set if not req.startswith(k)}
            requirements_set.add(v + "\n")
        except pkg_resources.DistributionNotFound:
            print(f"{k} is not installed.")

    with open(output_file, 'w') as output:
        for requirement in sorted(requirements_set):
            output.write(requirement)

def main():
    parser = argparse.ArgumentParser(description="Extract requirements from multiple directories using pipreqs and merge them into a single requirements file.")
    parser.add_argument("directories", nargs="+", help="List of directories to scan for requirements.")
    parser.add_argument("--output", "-o", default="requirements.txt", help="Path to the output requirements file. Default is 'requirements.txt' in the current directory.")
    
    args = parser.parse_args()
    directories = args.directories
    output_file = args.output
    temp_files = []

    try:
        for i, directory in enumerate(directories):
            temp_file = f"temp_requirements_{i}.txt"
            run_pipreqs(directory, temp_file)
            temp_files.append(temp_file)
        
        merge_requirements(temp_files, output_file)
        print(f"Combined requirements.txt has been created at {output_file}")
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    main()
