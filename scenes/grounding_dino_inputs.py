"""Grounding DINO image and BERT-uncased text inputs, shared with calibration."""

import unicodedata
from pathlib import Path

import cv2
import numpy as np

HEIGHT, WIDTH = 400, 599
MODEL = "groundingdino_swint_ogc_400x599_int8.onnx"
REVISION = "grounding-coco-val2017-ql-v3"
DEFAULT_PROMPT = "Horse. Clouds. Grasses. Sky. Hill."


class Tokenizer:
    """BERT uncased basic tokenization and greedy WordPiece (no downloads)."""

    def __init__(self, path):
        self.vocab = {word: i for i, word in enumerate(Path(path).read_text().splitlines())}

    def encode(self, caption, pad_to=None):
        caption = caption.strip().lower()
        if not caption:
            raise ValueError("Enter an English detection prompt")
        if not caption.endswith("."):
            caption += "."
        words, current = [], ""
        for char in unicodedata.normalize("NFD", caption):
            category = unicodedata.category(char)
            if category == "Mn":
                continue
            if category.startswith("C") and char not in "\t\n\r":
                continue
            punct = category.startswith("P") or (
                ord(char) < 128 and not char.isalnum() and not char.isspace()
            )
            chinese = 0x4E00 <= ord(char) <= 0x9FFF
            if char.isspace() or punct or chinese:
                if current:
                    words.append(current)
                    current = ""
                if punct or chinese:
                    words.append(char)
            else:
                current += char
        if current:
            words.append(current)
        tokens = ["[CLS]"]
        for word in words:
            pieces, start = [], 0
            while start < len(word) and len(word) <= 100:
                end = len(word)
                while end > start:
                    piece = ("##" if start else "") + word[start:end]
                    if piece in self.vocab:
                        break
                    end -= 1
                if end == start:
                    break
                pieces.append(piece)
                start = end
            tokens.extend(pieces if start == len(word) else ["[UNK]"])
        tokens.append("[SEP]")
        if len(tokens) > 256:
            raise ValueError("Prompt exceeds the model's 256-token limit; use fewer categories")
        ids = np.array([[self.vocab[t] for t in tokens]], dtype=np.int64)
        size = len(tokens)
        mask = np.eye(size, dtype=bool)[None]
        positions = np.zeros((1, size), dtype=np.int64)
        previous = 0
        for col, token in enumerate(tokens):
            if token not in ("[CLS]", "[SEP]", ".", "?"):
                continue
            if col not in (0, size - 1):
                mask[0, previous + 1 : col + 1, previous + 1 : col + 1] = True
                positions[0, previous + 1 : col + 1] = np.arange(col - previous)
            previous = col
        feed = dict(
            input_ids=ids,
            token_type_ids=np.zeros_like(ids),
            attention_mask=mask,
            position_ids=positions,
            text_token_mask=np.ones_like(ids, dtype=bool),
        )
        if pad_to is not None:
            if not size <= pad_to <= 256:
                raise ValueError("Padding length must fit the prompt and be at most 256")
            padding = pad_to - size
            for name, value in feed.items():
                if name == "attention_mask":
                    padded = np.eye(pad_to, dtype=bool)[None]
                    padded[:, :size, :size] = value
                    feed[name] = padded
                else:
                    feed[name] = np.pad(
                        value, ((0, 0), (0, padding)),
                        constant_values=self.vocab["[PAD]"] if name == "input_ids" else 0,
                    )
        return feed, tokens


def preprocess(image):
    """Aspect-preserving letterbox to the calibrated 400x599 input."""
    height, width = image.shape[:2]
    scale = min(WIDTH / width, HEIGHT / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (WIDTH - w) // 2, (HEIGHT - h) // 2
    rgb = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    canvas[top : top + h, left : left + w] = rgb.astype(np.float32) / 255
    normalized = (canvas - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), (left, top, w, h)


def detections(outputs, tokens, geometry, threshold=0.35, text_threshold=0.25):
    logits, boxes = outputs[:2]
    logits = logits[0, :, : len(tokens)]
    if not np.isfinite(logits).all() or not np.isfinite(boxes).all():
        raise RuntimeError("Non-finite Grounding DINO outputs")
    scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    left, top, w, h = geometry
    result = []
    for row, box in zip(scores, boxes[0]):
        if row.max() <= threshold:
            continue
        words = [
            t
            for t, s in zip(tokens, row)
            if s > text_threshold and t not in ("[CLS]", "[SEP]", ".", "?")
        ]
        label = " ".join(words).replace(" ##", "")
        xyxy = np.r_[box[:2] - box[2:] / 2, box[:2] + box[2:] / 2]
        xyxy = (xyxy * [WIDTH, HEIGHT, WIDTH, HEIGHT] - [left, top, left, top]) / [w, h, w, h]
        xyxy = np.clip(xyxy, 0, 1)
        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
            result.append(dict(label=label, score=float(row.max()), box=xyxy.tolist()))
    return result
