import argparse
import os
import re

from sleap import Labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input file.")
    parser.add_argument(
        "-o", "--output", default="", help="Path to output file (optional)."
    )
    parser.add_argument(
        "--format",
        default="slp",
        help="Output format. Default ('slp') is SLEAP dataset; "
        "'analysis' results in analysis.h5 file; "
        "'h5' or 'json' results in SLEAP dataset "
        "with specified file format.",
    )

    args = parser.parse_args()

    video_callback = Labels.make_video_callback([os.path.dirname(args.input_path)])
    try:
        labels = Labels.load_file(args.input_path, video_callback=video_callback)
    except TypeError:
        print("Input file isn't SLEAP dataset so attempting other importers...")
        from sleap.io.format import read

        labels = read(
            args.input_path,
            for_object="labels",
            as_format="*",
            video_callback=video_callback,
        )

    if args.format == "analysis":
        from sleap.info.write_tracking_h5 import main as write_analysis

        if args.output:
            output_path = args.output
        else:
            output_path = args.input_path
            output_path = re.sub("(\.json(\.zip)?|\.h5|\.slp)$", "", output_path)
            output_path = output_path + ".analysis.h5"

        write_analysis(labels, output_path=output_path, all_frames=True)

    elif args.output:
        print(f"Output SLEAP dataset: {args.output}")
        Labels.save_file(labels, args.output)

    elif args.format in ("slp", "h5", "json"):
        output_path = f"{args.input_path}.{args.format}"
        print(f"Output SLEAP dataset: {output_path}")
        Labels.save_file(labels, output_path)

    else:
        print("You didn't specify how to convert the file.")
        print(args)


if __name__ == "__main__":
    main()
