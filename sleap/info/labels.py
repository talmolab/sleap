from sleap.io.dataset import Labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    args = parser.parse_args()

    labels = Labels.load_json(args.data_path)

    print(f"Number of labeled frames: {len(labels)}")