import subprocess
import re

# Paths to the groundtruth file and output file
groundtruth_file = "/xinfolder/TUM/rgbd_dataset_freiburg3_sitting_rpy/groundtruth.txt"
output_file = "accuracy_results.txt"

# Command template
command_template = "evo_ape tum {} -vas {}"

# Open the output file for writing
with open(output_file, "w") as out_file:
    # Loop over CameraTrajectory files from 0 to 10
    for i in range(10,20):
        # Construct the current trajectory filename
        trajectory_file = f"CameraTrajectory_{i}.txt"

        # Construct the full command
        command = command_template.format(groundtruth_file, trajectory_file)

        try:
            # Run the command and capture the output
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout

            # Extract only the metrics section from the output
            metrics_start = output.find("       max	")
            if metrics_start != -1:
                metrics_section = output[metrics_start:].strip().splitlines()[:7]

                # Use regex to extract only the numeric values from each line
                metrics_values = [re.findall(r"[-+]?\d*\.\d+|\d+", line) for line in metrics_section]

                # Flatten the list of lists and join the numbers into a single line
                metrics_text = " ".join([value for values in metrics_values for value in values])
            else:
                metrics_text = "Metrics not found"

                # Write results to file
            out_file.write(metrics_text + "\n")
            print(f"Processed {trajectory_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error running command for {trajectory_file}: {e}")

print(f"All results have been collected in {output_file}")

