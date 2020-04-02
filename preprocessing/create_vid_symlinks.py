import os
import argparse


def create_vid_links(args):
    os.makedirs(args.symlinks_dir, exist_ok=True)

    for root, _, files in os.walk(args.video_dir):
        if len(files) > 0:
            for f in files:
                if f.endswith("MP4"):
                    source = os.path.join(root, f)
                    link = os.path.join(args.symlinks_dir, f)
                    if os.path.exists(link):
                        os.unlink(link)
                    os.symlink(source, link)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=str, help="Directory of videos")
    parser.add_argument(
        "symlinks_dir", type=str, help="Directory to save symlinks for EPIC"
    )

    args = parser.parse_args()

    create_vid_links(args)
