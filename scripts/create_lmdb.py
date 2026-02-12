import lmdb
import pickle
import os
from pathlib import Path
import os

"""
Build an LMDB database from PNG images. Images are stored as raw PNG bytes.
"""

splits = ["train", "val", "test"]
dst_main_dir = "DST_PATH/Phoenix_lmdb"
source_dir = "SOURCE_PATH/fullFrame-210x260px/"

for partition in splits:
    video_names_list = os.listdir(f"{source_dir}/{partition}")
    for each in video_names_list:
        dst_database = Path(f"{dst_main_dir}/{partition}/{each}")
        if not os.path.isdir(dst_database):
            print(dst_database)
            dst_database.mkdir(parents=True, exist_ok=True)
            n_bytes = 2 ** 40

            img_paths = list(Path(f"{source_dir}/{partition}/{each}").glob("**/*.png"))
            img_paths.sort(key=lambda p: int(p.stem.replace("images", "")))

            with lmdb.open(path=str(dst_database), map_size=n_bytes) as env:
                # Add the protocol to the database.
                with env.begin(write=True) as txn:
                    key = "protocol".encode("ascii")
                    value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
                    txn.put(key=key, value=value, dupdata=False)

                list_of_keys = []
                for i, pth in enumerate(img_paths):
                    img_name = int((pth.stem).split("images")[1])
                    folder_name = (str(pth.parents[0]).split('/')[-1])
                    key = pickle.dumps(i)
                    with env.begin(write=True) as txn:
                        with open(pth, mode="rb") as file:
                            txn.put(
                                key=key,
                                value=file.read(),
                                dupdata=False,
                            )
                    list_of_keys.append(key)

                with env.begin(write=True) as txn:
                    key = pickle.dumps("keys")
                    txn.put(
                        key=key,
                        value=pickle.dumps(sorted(list_of_keys)),
                        dupdata=False,
                    )