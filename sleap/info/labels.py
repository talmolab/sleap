import os

from sleap.io.dataset import Labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    args = parser.parse_args()

    def video_callback(video_list, new_paths=[os.path.dirname(args.data_path)]):
        # Check each video
        for video_item in video_list:
            if "backend" in video_item and "filename" in video_item["backend"]:
                current_filename = video_item["backend"]["filename"]
                # check if we can find video
                if not os.path.exists(current_filename):
                    is_found = False

                    current_basename = os.path.basename(current_filename)
                    # handle unix, windows, or mixed paths
                    if current_basename.find("/") > -1:
                        current_basename = current_basename.split("/")[-1]
                    if current_basename.find("\\") > -1:
                        current_basename = current_basename.split("\\")[-1]

                    # First see if we can find the file in another directory,
                    # and if not, prompt the user to find the file.

                    # We'll check in the current working directory, and if the user has
                    # already found any missing videos, check in the directory of those.
                    for path_dir in new_paths:
                        check_path = os.path.join(path_dir, current_basename)
                        if os.path.exists(check_path):
                            # we found the file in a different directory
                            video_item["backend"]["filename"] = check_path
                            is_found = True
                            break

    labels = Labels.load_json(args.data_path, video_callback=video_callback)

    print(f"Labeled frames: {len(labels)}")
    print(f"Tracks: {len(labels.tracks)}")

    print(f"Video files:")
    
    for vid in labels.videos:
        print(f"  {vid.filename}")
        first_idx = min((lf.frame_idx for lf in labels.find(vid)))
        last_idx = max((lf.frame_idx for lf in labels.find(vid)))

        print(f"    labeled from {first_idx} to {last_idx}")
