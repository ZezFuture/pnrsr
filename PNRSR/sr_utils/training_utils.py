import os
import cv2
import json
import torch
import random
import time
import numpy as np
import hashlib
from contextlib import contextmanager
from .realesrgan import RealESRGAN_degradation
from torch.utils.data import Dataset


class MyDataset_blind_plus(Dataset):
    def __init__(self, img_lst_dirs=None, config="params_realesrgan_seesr.yml",
                 deterministic_k=True, base_seed=0, resolution=512):
        self.data = []
        self.img_lst_dirs = img_lst_dirs if isinstance(img_lst_dirs, list) else [img_lst_dirs]
        self.deterministic_k = deterministic_k
        self.base_seed = base_seed
        self.degradation = RealESRGAN_degradation(config, device='cpu')
        self.resolution = resolution

        for img_dir in self.img_lst_dirs:
            self._collect_images(img_dir)

        self.data.sort()

        self.epoch = 0

    def _collect_images(self, root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append(os.path.join(root, file))

    def __len__(self):
        return len(self.data)

    def _random_crop(self, img, resolution):
        h, w, _ = img.shape
        if h <= resolution and w <= resolution:
            return img  

        startx = random.randint(0, w - resolution) if w > resolution else 0
        starty = random.randint(0, h - resolution) if h > resolution else 0
        return img[starty:starty + resolution, startx:startx + resolution]

    def _stable_seed(self, idx: int) -> int:
        s = f"{idx}-{self.epoch}-{self.base_seed}".encode("utf-8")
        return int.from_bytes(hashlib.md5(s).digest()[:4], byteorder="little", signed=False)

    def __getitem__(self, idx):
        target_filename = self.data[idx]
        target = cv2.imread(target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        h, w, _ = target.shape

        if self.deterministic_k:
            seed = self._stable_seed(idx)
            with self._temp_seed(seed):
                if h >= self.resolution and w >= self.resolution:
                    target = self._random_crop(target, self.resolution)
                else:
                    target = cv2.resize(target, (self.resolution, self.resolution), cv2.INTER_CUBIC)

                target, source = self.degradation.degrade_process(
                    np.asarray(target) / 255., resize_bak=True
                )
        else:
            if h >= self.resolution and w >= self.resolution:
                target = self._random_crop(target, self.resolution)
            else:
                target = cv2.resize(target, (self.resolution, self.resolution), cv2.INTER_CUBIC)

            target, source = self.degradation.degrade_process(
                np.asarray(target) / 255., resize_bak=True
            )

        source = (source.squeeze(0) * 2) - 1.0
        target = (target.squeeze(0) * 2) - 1.0
        return dict(output_pixel_values=target, conditioning_pixel_values=source)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @contextmanager
    def _temp_seed(self, seed: int):
        np_state = np.random.get_state()
        py_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.set_rng_state(torch_state)

    @staticmethod
    def collate_fn(examples):
        output_pixel_values = [example["output_pixel_values"] for example in examples]
        conditioning_pixel_values = [example["conditioning_pixel_values"] for example in examples]
        return output_pixel_values, conditioning_pixel_values

_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _atomic_write_text(path: str, text: str):
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        os.replace(tmp, path) 
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass

class MyDataset_blind_plus_withprompt(torch.utils.data.Dataset):

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        prompt_dir: str,
        deterministic_k: bool = False,
        base_seed: int = 0,
        resize_interp=cv2.INTER_CUBIC,
        use_index_cache: bool = True,
        index_cache_name: str = ".pairs_index.txt",
        sort_files: bool = False, 
    ):
        self.lr_dir = os.path.abspath(lr_dir)
        self.hr_dir = os.path.abspath(hr_dir)
        self.prompt_dir = os.path.abspath(prompt_dir)

        self.deterministic_k = deterministic_k
        self.base_seed = base_seed
        self.resize_interp = resize_interp

        self.epoch = 0

        self.index_cache_path = os.path.join(self.lr_dir, index_cache_name)
        self.rel_paths = self._load_or_build_index(
            use_index_cache=use_index_cache,
            sort_files=sort_files,
        )

    # -------------------- Index building (fast) --------------------
    def _scan_relpaths_scandir(self):
        
        rels = []
        stack = [(self.lr_dir, "")]  # (abs_dir, rel_prefix)

        while stack:
            abs_dir, rel_prefix = stack.pop()
            try:
                with os.scandir(abs_dir) as it:
                    for entry in it:
                        name = entry.name
                        if name.startswith("."):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append((entry.path, rel_prefix + name + "/"))
                        else:
                            low = name.lower()
                            if low.endswith(_IMG_EXTS):
                                rels.append(rel_prefix + name)
            except FileNotFoundError:
                continue

        return rels

    def _load_or_build_index(self, use_index_cache: bool, sort_files: bool):
        if use_index_cache and os.path.isfile(self.index_cache_path):
            with open(self.index_cache_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        if use_index_cache:
            rank = int(os.environ.get("RANK", "0"))
            if rank != 0:
                for _ in range(100):  
                    if os.path.isfile(self.index_cache_path):
                        with open(self.index_cache_path, "r", encoding="utf-8") as f:
                            return [line.strip() for line in f if line.strip()]
                    time.sleep(0.1)

        rels = self._scan_relpaths_scandir()
        if sort_files:
            rels.sort()

        if use_index_cache and (not os.path.isfile(self.index_cache_path)):
            try:
                _atomic_write_text(self.index_cache_path, "\n".join(rels) + "\n")
            except Exception:
                pass

        return rels

    def __len__(self):
        return len(self.rel_paths)

    def _stable_seed(self, idx: int) -> int:
        s = f"{idx}-{self.epoch}-{self.base_seed}".encode("utf-8")
        return int.from_bytes(hashlib.md5(s).digest()[:4], byteorder="little", signed=False)

    @contextmanager
    def _temp_seed(self, seed: int):
        np_state = np.random.get_state()
        py_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
            torch.set_rng_state(torch_state)

    def _read_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_tensor_minus1_1(self, img_rgb):
        x = img_rgb.astype(np.float32) / 127.5 - 1.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        return torch.from_numpy(x).float()

    def _read_prompt_fast(self, rel_img_path: str) -> str:
        stem = os.path.splitext(rel_img_path)[0]
        p_txt = os.path.join(self.prompt_dir, stem + ".txt")
        try:
            with open(p_txt, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass

        p_json = os.path.join(self.prompt_dir, stem + ".json")
        try:
            with open(p_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for k in ["prompt", "caption", "text", "description"]:
                    if k in obj and isinstance(obj[k], str):
                        return obj[k].strip()
                return json.dumps(obj, ensure_ascii=False)
            if isinstance(obj, str):
                return obj.strip()
            return str(obj)
        except Exception:
            return ""

    # -------------------- Dataset API --------------------
    def __getitem__(self, idx):
        rel = self.rel_paths[idx]

        lr_path = os.path.join(self.lr_dir, rel)
        hr_path = os.path.join(self.hr_dir, rel)

        lr = self._read_rgb(lr_path)
        hr = self._read_rgb(hr_path)

        if lr.shape[:2] != hr.shape[:2]:
            h, w = hr.shape[:2]
            lr = cv2.resize(lr, (w, h), interpolation=self.resize_interp)

        if self.deterministic_k:
            seed = self._stable_seed(idx)
            with self._temp_seed(seed):
                pass
        else:
            seed = int.from_bytes(os.urandom(4), "little")

        prompt = self._read_prompt_fast(rel)

        return dict(
            output_pixel_values=self._to_tensor_minus1_1(hr),
            conditioning_pixel_values=self._to_tensor_minus1_1(lr),
            seeds=seed,
            prompt=prompt,
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @staticmethod
    def collate_fn(examples):
        output_pixel_values = [ex["output_pixel_values"] for ex in examples]
        conditioning_pixel_values = [ex["conditioning_pixel_values"] for ex in examples]
        seeds = [ex["seeds"] for ex in examples]
        prompts = [ex.get("prompt", "") for ex in examples]
        return output_pixel_values, conditioning_pixel_values, prompts,seeds
    