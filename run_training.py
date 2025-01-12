import os
import subprocess

# Define variables
maxeps = 150
f = 9
# cuda_devices = "2,3,4,5,6,7"

# python main.py --model res18 -b 8 --resume 064.ckpt --save-dir res18/retrft969/ --epochs 150 --config config_training9
os.chdir("detector")
# Step 2: Loop through epochs
for i in range(1, maxeps + 1):
    print(f"Processing epoch {i}")

    # Construct checkpoint path based on epoch number
    if i < 10:
        checkpoint = f"results/res18/retrft96{f}/00{i}.ckpt"
    elif i < 100:
        checkpoint = f"results/res18/retrft96{f}/0{i}.ckpt"
    elif i < 1000:
        checkpoint = f"results/res18/retrft96{f}/{i}.ckpt"
    else:
        print("Unhandled case")
        continue

    # Step 3: Run the test command
    test_cmd = f"python main.py --model res18 -b 8 --resume {checkpoint} --test 1 --save-dir res18/retrft96{f}/ --config config_training{f}"
    subprocess.run(test_cmd, shell=True, check=True)

    # Step 4: Create validation directory if it doesn't exist
    val_dir = f"results/res18/retrft96{f}/val{i}/"
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Step 5: Move .npy files to validation directory
    bbox_dir = f"results/res18/retrft96{f}/bbox/"
    for file_name in os.listdir(bbox_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(bbox_dir, file_name)
            os.rename(file_path, os.path.join(val_dir, file_name))