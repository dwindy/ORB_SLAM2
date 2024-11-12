#in the future. check the difference between two lines of grountruth. apply the difference to the input file.

import sys
import random

def process_files(groundtruth_file, second_file, output_file, disturbance_range):
    try:
        # Read the groundtruth and second files
        with open(groundtruth_file, 'r') as gt_file, open(second_file, 'r') as sec_file:
            gt_lines = gt_file.readlines()[3:]  
            sec_lines = sec_file.readlines()

        # Process each line in the second file
        counter = 0
        modified_lines = []
        for sec_line in sec_lines:
            counter = counter + 1
            sec_elements = sec_line.strip().split()
            timestamp_sec = float(sec_elements[0])  # Timestamp from second file
            
            # Find the closest timestamp in the groundtruth file
            closest_gt_line = None
            closest_timestamp_diff = float('inf')
            for gt_line in gt_lines:
                gt_elements = gt_line.strip().split()
                timestamp_gt = float(gt_elements[0])  # Timestamp from groundtruth file

                # Calculate the absolute difference between the timestamps
                timestamp_diff = abs(timestamp_gt - timestamp_sec)
                if timestamp_diff < closest_timestamp_diff:
                    closest_timestamp_diff = timestamp_diff
                    closest_gt_line = gt_line

            # If a closest line is found, process it
            if closest_gt_line:
                gt_elements = closest_gt_line.strip().split()
                # Extract quaternion and translation from the groundtruth line
                #gt_quaternion = list(map(float, gt_elements[4:8]))
                #gt_translation = list(map(float, gt_elements[1:4]))
                gt_quaternion = list(map(float, gt_elements[1:4]))
                gt_translation = list(map(float, gt_elements[4:8]))

                # Apply disturbances (random value within the specified range)
                disturbed_quaternion = [
                    gt_q + random.uniform(-disturbance_range, disturbance_range)
                    for gt_q in gt_quaternion
                ]
                disturbed_translation = [
                    gt_t + random.uniform(-disturbance_range, disturbance_range)
                    for gt_t in gt_translation
                ]

                # Construct the modified line using the timestamp from the second file
                modified_line = f"{timestamp_sec} " + " ".join(map(str, disturbed_quaternion + disturbed_translation)) + "\n"
                modified_lines.append(modified_line)

            if counter % 1000 == 0:
                print(f"counter: {counter}")

        # Write the modified lines to the output file
        with open(output_file, 'w') as out_file:
            out_file.writelines(modified_lines)

        print(f"Output file '{output_file}' has been created successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Check if filenames and disturbance range were provided as command-line arguments
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <groundtruth_file> <second_file> <output_file> <disturbance_range>")
    else:
        groundtruth_file = sys.argv[1]
        second_file = sys.argv[2]
        output_file = sys.argv[3]
        disturbance_range = float(sys.argv[4])
        process_files(groundtruth_file, second_file, output_file, disturbance_range)

