"""
Helper script to update Google Drive download links in classification.py

Instructions:
1. Go to your Google Drive folder: https://drive.google.com/drive/folders/1uSlAS4t8_1AkUZmX0q_61jf5_5-TpYtZ
2. For each .h5 file:
   - Right click > Get link > Copy link
   - The link will be like: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
   - Extract the FILE_ID from the URL
3. Run this script and paste the FILE_IDs when prompted

Alternatively, you can directly edit this script with the file IDs and run it.
"""

import re


def update_drive_links():
    """Update Google Drive links in classification.py"""

    # Dictionary to store file IDs
    # You can either fill this in manually or run the script interactively
    file_ids = {
        'EfficientNetB3': '',  # Paste FILE_ID here or leave empty to be prompted
        'InceptionV3': '',
        'InceptionResNetV2': ''
    }

    print("=" * 60)
    print("Google Drive Link Updater")
    print("=" * 60)
    print("\nCurrent Drive folder: https://drive.google.com/drive/folders/1uSlAS4t8_1AkUZmX0q_61jf5_5-TpYtZ\n")

    # Prompt for file IDs if not provided
    for model in file_ids.keys():
        if not file_ids[model]:
            print(f"\n{model}.h5:")
            print("  1. Right-click the file in Google Drive")
            print("  2. Click 'Get link' > Copy link")
            print("  3. Paste the full URL or just the FILE_ID")
            file_input = input(f"  Enter link/ID for {model}: ").strip()

            # Extract file ID from URL if full URL was provided
            if 'drive.google.com' in file_input:
                match = re.search(r'/d/([a-zA-Z0-9_-]+)', file_input)
                if match:
                    file_ids[model] = match.group(1)
                else:
                    print("  ⚠ Could not extract ID from URL, using as-is")
                    file_ids[model] = file_input
            else:
                file_ids[model] = file_input

    # Validate file IDs
    print("\n" + "=" * 60)
    print("File IDs to be used:")
    print("=" * 60)
    for model, file_id in file_ids.items():
        if file_id:
            print(f"✓ {model}: {file_id}")
        else:
            print(f"✗ {model}: MISSING")

    if not all(file_ids.values()):
        print("\n⚠ Some file IDs are missing. Please provide all IDs.")
        return

    confirm = input("\nProceed with update? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Read current classification.py
    with open('classification.py', 'r') as f:
        content = f.read()

    # Update URLs
    for model, file_id in file_ids.items():
        # Pattern to match the URL line for this model
        pattern = rf"('{model}':\s*\{{[^}}]*'url':\s*)'[^']*'"
        replacement = rf"\1'https://drive.google.com/uc?id={file_id}'"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write updated file
    with open('classification.py', 'w') as f:
        f.write(content)

    print("\n✓ classification.py updated successfully!")
    print("\nUpdated URLs:")
    for model, file_id in file_ids.items():
        print(f"  {model}: https://drive.google.com/uc?id={file_id}")


def show_manual_instructions():
    """Show manual update instructions"""
    print("\n" + "=" * 60)
    print("Manual Update Instructions")
    print("=" * 60)
    print("""
If you prefer to update manually:

1. Open classification.py

2. Find the MODEL_CONFIGS dictionary (around line 29)

3. For each model, replace the 'url' value:

   Before:
   'EfficientNetB3': {
       'url': 'https://drive.google.com/uc?id=OLD_FILE_ID',
       ...
   }

   After:
   'EfficientNetB3': {
       'url': 'https://drive.google.com/uc?id=YOUR_NEW_FILE_ID',
       ...
   }

4. Get YOUR_NEW_FILE_ID from your Drive folder:
   - Go to: https://drive.google.com/drive/folders/1uSlAS4t8_1AkUZmX0q_61jf5_5-TpYtZ
   - Right-click each .h5 file > Get link
   - Extract the ID from the URL (the part after '/d/')

5. Save classification.py
""")


if __name__ == '__main__':
    import sys

    print(__doc__)

    choice = input("\nChoose option:\n  1. Interactive update\n  2. Show manual instructions\n  3. Exit\n\nChoice (1/2/3): ").strip()

    if choice == '1':
        update_drive_links()
    elif choice == '2':
        show_manual_instructions()
    else:
        print("Exiting.")
