from argparse import ArgumentParser
import shutil
from pathlib import Path
import cv2
from PIL import Image
from time import time
import io
import lmdb
import pickle
import os
from tqdm import tqdm

"""
Build an LMDB database. Resize, save as jpeg compressed.
Adapted from https://github.com/ryanwongsa/Sign2GPT/blob/main/scripts/csldaily/image_lmdb_creator.py
"""

def main(params):
    video_names = os.listdir(params.data_dir)[params.start_ind:params.end_ind]
    for video_name in tqdm(video_names):
        video_name = video_name.replace(".mp4", "")
        data_dir = Path(params.data_dir)
        save_dir = Path(params.save_dir + "/" + video_name + ".lmdb")

        video_path = str(data_dir / f"{video_name}.mp4")

        n_bytes = 2**40

        tmp_dir = Path("/tmp") / f"TEMP_{time()}"
        env = lmdb.open(path=str(tmp_dir), map_size=n_bytes)
        txn = env.begin(write=True)

        cap = cv2.VideoCapture(video_path)

        if save_dir.exists() and save_dir.is_dir():
            print("exist")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        ind = 0
        counter = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            ret, frame = cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crop to square
            # h, w, _ = frame.shape
            # min_dim = min(h, w)
            # start_x = (w - min_dim) // 2
            # start_y = (h - min_dim) // 2
            # cropped_frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

            # resize
            img = Image.fromarray(frame).resize((256, 256))
            # img.save(f"{video_name}_{counter}.png")  # check image
            temp = io.BytesIO()
            img.save(temp, format="jpeg")
            temp.seek(0)
            txn.put(
                key=f"{ind}".encode("ascii"),
                value=temp.read(),
                dupdata=False,
            )
            ind += 1
            counter += 1

            if counter % 123 == 0 and counter != 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.put(
            key=f"details".encode("ascii"),
            value=pickle.dumps({"num_frames": ind, "video_name": video_name}, protocol=4),
            dupdata=False,
        )
        txn.commit()

        env.close()

        if save_dir.exists():
            shutil.rmtree(save_dir)
        shutil.move(f"{tmp_dir}", f"{save_dir}")

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save lmdb files."
    )
    parser.add_argument(
        "--start_ind",
        type=int,
        default="0",
        help="Start frame index of the video segment. "
             "If not provided, the video is used from the first frame (index 0)."
    )
    parser.add_argument(
        "--end_ind",
        type=int,
        default="1000",
        help="End frame index of the video segment. "
             "If not provided, the video is used until the last frame."
    )

    params, unknown = parser.parse_known_args()
    main(params)
