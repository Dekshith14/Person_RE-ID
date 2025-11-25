import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torchreid

# ===========================
# 1. MODEL DEFS (from notebook)
# ===========================

class BasicConv2d(nn.Module):
    """A simple 2D Convolutional block with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GaitCNNLSTM(nn.Module):
    """
    CNN-LSTM architecture.
    (Same as training notebook)
    """
    def __init__(self, embedding_dim=256, num_subjects=74, lstm_hidden_dim=512):
        super(GaitCNNLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_subjects = num_subjects
        self.lstm_hidden_dim = lstm_hidden_dim

        # --- CNN Backbone ---
        self.conv1 = BasicConv2d(1, 32, 5, 1, 2)
        self.conv2 = BasicConv2d(32, 32, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = BasicConv2d(32, 64, 3, 1, 1)
        self.conv4 = BasicConv2d(64, 64, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv5 = BasicConv2d(64, 128, 3, 1, 1)
        self.conv6 = BasicConv2d(128, 128, 3, 1, 1)
        self.cnn_feature_size = 128 * 16 * 16

        # --- LSTM Layer ---
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_size,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # --- Head ---
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim, self.num_subjects)

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.shape
        x = x.view(batch_size * seq_len, 1, 64, 64)

        x = self.conv1(x); x = self.conv2(x); x = self.maxpool1(x)
        x = self.conv3(x); x = self.conv4(x); x = self.maxpool2(x)
        x = self.conv5(x); x = self.conv6(x)

        x = x.view(batch_size * seq_len, -1)
        x = x.view(batch_size, seq_len, self.cnn_feature_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.squeeze(0)

        embedding = self.fc1(x)
        logits = self.classifier(embedding)
        return logits, embedding


# ===========================
# 2. GLOBAL CONFIG
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAIT_EMBEDDING_DIM = 256
NUM_SUBJECTS = 74
SEQ_LEN = 30
REID_THRESHOLD = 1.5  # same as notebook

# paths relative to project root
GAIT_MODEL_PATH = os.path.join("models", "my_gait_cnnlstm.pth")
GALLERY_PATH = os.path.join("gallery", "my_known_gallery.pth")

# globals to cache models
_gait_model = None
_appearance_model = None
_yolo_seg_model = None
_pose_model = None
_gait_transform = None
_appearance_transform = None
_gallery_of_known_people = None


# ===========================
# 3. HELPERS
# ===========================

def get_body_ratios(kpts):
    """
    Same as your notebook: compute torso/leg ratios from keypoints.
    """
    try:
        nose_y = kpts[0, 1]
        shoulder_y = (kpts[5, 1] + kpts[6, 1]) / 2.0
        hip_y = (kpts[11, 1] + kpts[12, 1]) / 2.0
        ankle_y = (kpts[15, 1] + kpts[16, 1]) / 2.0

        total_height = ankle_y - nose_y
        torso_height = hip_y - shoulder_y
        leg_height = ankle_y - hip_y

        if total_height > 1 and leg_height > 1:
            ratio_1 = torso_height / total_height
            ratio_2 = leg_height / total_height
            return torch.tensor([ratio_1, ratio_2], dtype=torch.float32).to(device)
    except Exception:
        pass
    return None


def _load_models_and_gallery():
    """
    Lazily load all heavy models and the gallery once.
    """
    global _gait_model, _appearance_model, _yolo_seg_model, _pose_model
    global _gait_transform, _appearance_transform, _gallery_of_known_people

    if _gait_model is not None:
        # already loaded
        return

    print(f"[INIT] Using device: {device}")

    # --- Gait model ---
    print("[INIT] Loading GaitCNNLSTM...")
    gait_model = GaitCNNLSTM(GAIT_EMBEDDING_DIM, NUM_SUBJECTS).to(device)
    gait_model.load_state_dict(torch.load(GAIT_MODEL_PATH, map_location=device))
    gait_model.eval()
    _gait_model = gait_model
    print("[INIT] GaitCNNLSTM loaded.")

    # --- Appearance model (OSNet) ---
    print("[INIT] Loading OSNet (Torchreid)...")
    appearance_model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        pretrained=True
    )
    appearance_model = appearance_model.to(device)
    appearance_model.eval()
    _appearance_model = appearance_model
    print("[INIT] OSNet loaded.")

    # --- YOLOv8-Seg for tracking ---
    print("[INIT] Loading YOLOv8-Seg...")
    yolo_seg_model = YOLO("yolov8n-seg.pt")
    yolo_seg_model.to(device)
    _yolo_seg_model = yolo_seg_model
    print("[INIT] YOLOv8-Seg loaded.")

    # --- YOLOv8-Pose ---
    print("[INIT] Loading YOLOv8-Pose...")
    pose_model = YOLO("yolov8n-pose.pt")
    pose_model.to(device)
    _pose_model = pose_model
    print("[INIT] YOLOv8-Pose loaded.")

    # --- transforms ---
    _gait_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    _appearance_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- gallery ---
    if os.path.exists(GALLERY_PATH):
        print(f"[INIT] Loading gallery from {GALLERY_PATH}...")
        gallery_of_known_people = torch.load(GALLERY_PATH, map_location=device)
        _gallery_of_known_people = gallery_of_known_people
        print(f"[INIT] Loaded {len(gallery_of_known_people)} known people:",
              list(gallery_of_known_people.keys()))
    else:
        print(f"[WARN] Gallery file not found at {GALLERY_PATH}. "
              f"Unknown people will just be 'Track-X'.")
        _gallery_of_known_people = {}


# ===========================
# 4. MAIN PIPELINE FUNCTION
# ===========================

def run_reid(input_video_path: str, output_video_path: str) -> str:
    """
    Run your 3-way fused person Re-ID pipeline on the input video,
    save the processed video to output_video_path, and return that path.
    """
    _load_models_and_gallery()

    gait_model = _gait_model
    appearance_model = _appearance_model
    yolo_seg_model = _yolo_seg_model
    pose_model = _pose_model
    gait_transform = _gait_transform
    appearance_transform = _appearance_transform
    gallery_of_known_people = _gallery_of_known_people

    # --- Open video ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # --- Data structures from notebook ---
    tracked_gait_sequences = {}
    tracked_appearance_crops = {}
    tracked_body_ratios = {}
    track_id_to_person_id = {}  # {track_id: (person_id, match_distance)}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_seg_model.track(frame, persist=True, classes=0, verbose=False)
            frame_draw_info = []

            if results and results[0].masks is not None and results[0].boxes.id is not None:
                masks = results[0].masks.data
                boxes = results[0].boxes.data
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for mask_tensor, box, track_id in zip(masks, boxes, track_ids):
                    x1, y1, x2, y2 = [int(i) for i in box[:4]]

                    # --- GAIT ---
                    mask_np = mask_tensor.cpu().numpy() * 255
                    mask_pil = Image.fromarray(mask_np).convert("L")
                    silhouette_tensor = gait_transform(mask_pil).to(device)

                    # --- APPEARANCE ---
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        continue
                    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    appearance_tensor = appearance_transform(crop_pil).to(device)

                    # --- POSE / body ratios ---
                    pose_results = pose_model(crop_img, verbose=False)
                    body_ratio_tensor = None
                    if len(pose_results[0].keypoints.data) > 0:
                        kpts = pose_results[0].keypoints.data[0].cpu().numpy()[:, :2]
                        body_ratio_tensor = get_body_ratios(kpts)

                    # store features
                    if track_id not in tracked_gait_sequences:
                        tracked_gait_sequences[track_id] = []
                        tracked_appearance_crops[track_id] = []
                        tracked_body_ratios[track_id] = []

                    tracked_gait_sequences[track_id].append(silhouette_tensor)
                    tracked_appearance_crops[track_id].append(appearance_tensor)
                    if body_ratio_tensor is not None:
                        tracked_body_ratios[track_id].append(body_ratio_tensor)

                    # if we have full sequence, run re-id
                    if len(tracked_gait_sequences[track_id]) == SEQ_LEN:
                        gait_sequence = torch.stack(tracked_gait_sequences[track_id], dim=0).unsqueeze(0)
                        appearance_sequence = torch.stack(tracked_appearance_crops[track_id], dim=0)

                        if len(tracked_body_ratios[track_id]) > 0:
                            ratio_sequence = torch.stack(tracked_body_ratios[track_id], dim=0)
                        else:
                            ratio_sequence = torch.zeros((SEQ_LEN, 2)).to(device)

                        with torch.no_grad():
                            _, gait_embedding = gait_model(gait_sequence)
                            appearance_embeddings = appearance_model(appearance_sequence)
                            appearance_embedding = torch.mean(appearance_embeddings, dim=0, keepdim=True)
                            body_ratio_embedding = torch.mean(ratio_sequence, dim=0, keepdim=True)

                            gait_embedding = F.normalize(gait_embedding, p=2, dim=1)
                            appearance_embedding = F.normalize(appearance_embedding, p=2, dim=1)
                            body_ratio_embedding = F.normalize(body_ratio_embedding, p=2, dim=1)

                            final_embedding = torch.cat(
                                (gait_embedding, appearance_embedding, body_ratio_embedding),
                                dim=1
                            )

                        # match with gallery
                        is_new_person = True
                        matched_id = None
                        min_distance = float("inf")

                        if len(gallery_of_known_people) > 0:
                            for person_name, known_embedding in gallery_of_known_people.items():
                                distance = torch.cdist(
                                    final_embedding,
                                    known_embedding.to(device)
                                ).item()
                                if distance < min_distance:
                                    min_distance = distance
                                    matched_id = person_name

                        if min_distance < REID_THRESHOLD:
                            is_new_person = False

                        if is_new_person:
                            track_id_to_person_id[track_id] = (f"Track-{track_id}", min_distance)
                        else:
                            track_id_to_person_id[track_id] = (matched_id, min_distance)

                        tracked_gait_sequences[track_id] = []
                        tracked_appearance_crops[track_id] = []
                        tracked_body_ratios[track_id] = []

                    # prepare info for drawing
                    person_id_mapping = track_id_to_person_id.get(track_id)
                    if person_id_mapping:
                        display_id = person_id_mapping[0]
                        match_distance = person_id_mapping[1]
                    else:
                        display_id = f"Track-{track_id}"
                        match_distance = float("inf")

                    frame_draw_info.append({
                        "box": box,
                        "display_id": display_id,
                        "track_id": track_id,
                        "match_distance": match_distance
                    })

                # --- conflict resolution (same as notebook) ---
                final_draw_list = []
                person_id_assignments = {}

                for info in frame_draw_info:
                    display_id = info["display_id"]
                    match_distance = info["match_distance"]
                    is_known = display_id in gallery_of_known_people

                    if is_known:
                        if display_id not in person_id_assignments:
                            person_id_assignments[display_id] = (match_distance, info)
                        else:
                            current_best_distance, _ = person_id_assignments[display_id]
                            if match_distance < current_best_distance:
                                _, old_info = person_id_assignments.pop(display_id)
                                old_info["display_id"] = f"Track-{old_info['track_id']}"
                                final_draw_list.append(old_info)
                                person_id_assignments[display_id] = (match_distance, info)
                            else:
                                info["display_id"] = f"Track-{info['track_id']}"
                                final_draw_list.append(info)
                    else:
                        final_draw_list.append(info)

                for best_distance, info in person_id_assignments.values():
                    final_draw_list.append(info)

                # --- draw boxes ---
                for info in final_draw_list:
                    box = info["box"]
                    display_id = info["display_id"]
                    x1, y1, x2, y2 = [int(i) for i in box[:4]]

                    is_known = display_id in gallery_of_known_people
                    color = (0, 255, 0) if is_known else (0, 0, 255)
                    label = f"{display_id}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[DONE] Video processing finished. Output saved to: {output_video_path}")

    return output_video_path
