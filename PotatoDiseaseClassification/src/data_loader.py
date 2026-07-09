import os

def explore_dataset(base_path):

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    total = 0

    classes = sorted(
    folder
    for folder in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, folder))
)

    for cls in classes:

        folder = os.path.join(base_path, cls)

        if os.path.isdir(folder):

            count = len(os.listdir(folder))

            total += count

            print(f"{cls:<30}{count}")

    print("\nTotal Images :", total)

    return classes