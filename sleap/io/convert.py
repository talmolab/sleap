"""
Command line utility for converting between various dataset formats.

Reads:

* SLEAP dataset in .slp, .h5, .json, or .json.zip file
* SLEAP "analysis" file in .h5 format
* LEAP dataset in .mat file
* DeepLabCut dataset in .yaml or .csv file
* DeepPoseKit dataset in .h5 file
* COCO keypoints dataset in .json file

Writes:

* SLEAP dataset (defaults to .slp if no extension specified)
* SLEAP "analysis" file (.h5)

You don't need to specify the input format; this will be automatically detected.

If you don't specify an output path, then by default we will convert to a .slp
dataset file and save it at `<input path>.slp`.

Analysis HDF5:

If you want to export an "analysis" h5 file, use `--format analysis`. If no
output path is specified, the default is `<input path>.analysis.h5`.

The analysis HDF5 file has these datasets:

* "track_occupancy"    (shape: tracks * frames)
* "tracks"             (shape: frames * nodes * 2 * tracks)
* "track_names"        (shape: tracks)
* "node_names"         (shape: nodes)

Note: the datasets are stored column-major as expected by MATLAB.
This means that if you're working with the file in Python you may want to
first transpose the datasets so they matche the shapes described above.

"""

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
    parser.add_argument(
        "--video", default="", help="Path to video (if needed for conversion)."
    )

    args = parser.parse_args()

    video_callback = Labels.make_video_callback([os.path.dirname(args.input_path)])
    try:
        labels = Labels.load_file(args.input_path, video_search=video_callback)
    except TypeError:
        print("Input file isn't SLEAP dataset so attempting other importers...")
        from sleap.io.format import read

        video_path = args.video if args.video else None

        labels = read(
            args.input_path,
            for_object="labels",
            as_format="*",
            video_search=video_callback,
            video=video_path,
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
