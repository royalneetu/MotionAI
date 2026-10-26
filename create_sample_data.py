"""
Synthetic Human Motion Dataset Generator for Human Activity Recognition.
Generates realistic human silhouette motion patterns and saves them as MP4 video clips
for Walking, Running, Sitting, Standing, Lying_Down, and Falling.
Enables immediate training, evaluation, and interactive dashboard testing without massive external downloads.
"""

import os
import cv2
import numpy as np
import math

CLASSES = ["Walking", "Running", "Sitting", "Standing", "Lying_Down", "Falling"]
WIDTH = 128
HEIGHT = 128
FPS = 15
CLIP_SECONDS = 2.0
TOTAL_FRAMES = int(FPS * CLIP_SECONDS)  # 30 frames


def draw_human(frame, x, y, scale=1.0, pose="standing", angle=0.0, leg_phase=0.0):
    """
    Renders an articulated human figure with head, torso, arms, and legs.
    """
    color = (235, 235, 235)  # Soft white
    joint_color = (0, 165, 255)  # Orange accent
    thickness = max(2, int(2 * scale))

    # Center coordinates
    cx, cy = int(x), int(y)

    if pose == "standing":
        # Head
        head_radius = int(8 * scale)
        cv2.circle(frame, (cx, cy - int(24 * scale)), head_radius, color, -1)
        # Torso
        cv2.line(frame, (cx, cy - int(16 * scale)), (cx, cy + int(10 * scale)), color, thickness)
        # Arms
        arm_swing = math.sin(leg_phase) * 6 * scale
        cv2.line(frame, (cx, cy - int(12 * scale)), (cx - int(10 * scale), cy + int(arm_swing)), color, thickness)
        cv2.line(frame, (cx, cy - int(12 * scale)), (cx + int(10 * scale), cy - int(arm_swing)), color, thickness)
        # Legs
        cv2.line(frame, (cx, cy + int(10 * scale)), (cx - int(6 * scale), cy + int(32 * scale)), color, thickness)
        cv2.line(frame, (cx, cy + int(10 * scale)), (cx + int(6 * scale), cy + int(32 * scale)), color, thickness)

    elif pose == "walking":
        head_radius = int(8 * scale)
        cv2.circle(frame, (cx, cy - int(24 * scale)), head_radius, color, -1)
        # Torso
        cv2.line(frame, (cx, cy - int(16 * scale)), (cx, cy + int(10 * scale)), color, thickness)
        # Arms swing opposite to legs
        swing = math.sin(leg_phase) * 12 * scale
        cv2.line(frame, (cx, cy - int(12 * scale)), (cx + int(swing), cy), color, thickness)
        cv2.line(frame, (cx, cy - int(12 * scale)), (cx - int(swing), cy), color, thickness)
        # Stride legs
        stride = math.sin(leg_phase) * 14 * scale
        cv2.line(frame, (cx, cy + int(10 * scale)), (cx - int(stride), cy + int(32 * scale)), color, thickness)
        cv2.line(frame, (cx, cy + int(10 * scale)), (cx + int(stride), cy + int(32 * scale)), color, thickness)

    elif pose == "running":
        head_radius = int(8 * scale)
        # Forward lean for running
        lean_x = cx + int(6 * scale)
        cv2.circle(frame, (lean_x, cy - int(24 * scale)), head_radius, color, -1)
        cv2.line(frame, (lean_x, cy - int(16 * scale)), (cx, cy + int(8 * scale)), color, thickness)
        # High arm swing
        swing = math.sin(leg_phase) * 18 * scale
        cv2.line(frame, (lean_x, cy - int(12 * scale)), (cx + int(swing), cy - int(4 * scale)), color, thickness)
        cv2.line(frame, (lean_x, cy - int(12 * scale)), (cx - int(swing), cy - int(4 * scale)), color, thickness)
        # Wide stride
        stride = math.sin(leg_phase) * 22 * scale
        knee_lift = abs(math.cos(leg_phase)) * 8 * scale
        cv2.line(frame, (cx, cy + int(8 * scale)), (cx - int(stride), cy + int(30 * scale) - int(knee_lift)), color, thickness)
        cv2.line(frame, (cx, cy + int(8 * scale)), (cx + int(stride), cy + int(30 * scale)), color, thickness)

    elif pose == "sitting":
        head_radius = int(8 * scale)
        cv2.circle(frame, (cx, cy - int(14 * scale)), head_radius, color, -1)
        # Bent torso & chair
        cv2.line(frame, (cx, cy - int(6 * scale)), (cx, cy + int(14 * scale)), color, thickness)
        # Thigh horizontal
        cv2.line(frame, (cx, cy + int(14 * scale)), (cx + int(14 * scale), cy + int(14 * scale)), color, thickness)
        # Lower leg vertical down
        cv2.line(frame, (cx + int(14 * scale), cy + int(14 * scale)), (cx + int(14 * scale), cy + int(32 * scale)), color, thickness)
        # Hands resting on knees
        cv2.line(frame, (cx, cy - int(4 * scale)), (cx + int(10 * scale), cy + int(12 * scale)), color, thickness)

    elif pose == "lying_down":
        head_radius = int(8 * scale)
        # Horizontal head
        cv2.circle(frame, (cx - int(26 * scale), cy), head_radius, color, -1)
        # Horizontal torso
        cv2.line(frame, (cx - int(18 * scale), cy), (cx + int(10 * scale), cy), color, thickness)
        # Horizontal legs
        cv2.line(frame, (cx + int(10 * scale), cy), (cx + int(34 * scale), cy), color, thickness)
        # Arms beside body
        cv2.line(frame, (cx - int(12 * scale), cy - int(4 * scale)), (cx + int(12 * scale), cy - int(4 * scale)), color, thickness)

    elif pose == "falling":
        # Rotated collapsing body
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        def rot_pt(dx, dy):
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            return int(cx + rx), int(cy + ry)

        h_pt = rot_pt(0, -int(24 * scale))
        cv2.circle(frame, h_pt, int(8 * scale), joint_color, -1)
        t_start = rot_pt(0, -int(16 * scale))
        t_end = rot_pt(0, int(10 * scale))
        cv2.line(frame, t_start, t_end, color, thickness)

        # Flailing limbs during fall
        a_left = rot_pt(-int(16 * scale), -int(10 * scale))
        a_right = rot_pt(int(16 * scale), -int(14 * scale))
        cv2.line(frame, t_start, a_left, color, thickness)
        cv2.line(frame, t_start, a_right, color, thickness)

        leg_left = rot_pt(-int(12 * scale), int(30 * scale))
        leg_right = rot_pt(int(12 * scale), int(28 * scale))
        cv2.line(frame, t_end, leg_left, color, thickness)
        cv2.line(frame, t_end, leg_right, color, thickness)


def generate_video_clip(activity: str, output_path: str, seed: int = 42):
    """
    Renders a 30-frame synthetic motion video sequence for a specific activity class.
    """
    np.random.seed(seed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    start_x = np.random.uniform(25, 45) if "Walking" in activity or "Running" in activity else WIDTH // 2
    y_base = HEIGHT - 36
    scale = np.random.uniform(0.9, 1.1)

    for i in range(TOTAL_FRAMES):
        # Room background with floor guideline
        frame = np.full((HEIGHT, WIDTH, 3), 32, dtype=np.uint8)  # Dark room background
        cv2.line(frame, (0, HEIGHT - 12), (WIDTH, HEIGHT - 12), (55, 55, 55), 1)  # Floor line

        # Subtle background gradient noise for realism
        noise = np.random.randint(-4, 4, (HEIGHT, WIDTH, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        t = i / TOTAL_FRAMES

        if activity == "Walking":
            cur_x = (start_x + i * 2.2) % (WIDTH - 20)
            cur_y = y_base + math.sin(i * 0.8) * 1.5
            leg_phase = i * 0.7
            draw_human(frame, cur_x, cur_y, scale=scale, pose="walking", leg_phase=leg_phase)

        elif activity == "Running":
            cur_x = (start_x + i * 4.8) % (WIDTH - 20)
            cur_y = y_base + math.sin(i * 1.4) * 3.0
            leg_phase = i * 1.3
            draw_human(frame, cur_x, cur_y, scale=scale, pose="running", leg_phase=leg_phase)

        elif activity == "Sitting":
            cur_x = WIDTH // 2 + np.random.uniform(-1, 1)
            # Starts standing, then lowers down to chair height
            if i < 8:
                cur_y = y_base
                draw_human(frame, cur_x, cur_y, scale=scale, pose="standing")
            else:
                progress = min(1.0, (i - 8) / 10.0)
                cur_y = y_base + progress * 10
                draw_human(frame, cur_x, cur_y, scale=scale, pose="sitting")

        elif activity == "Standing":
            cur_x = WIDTH // 2 + math.sin(i * 0.1) * 1.0
            cur_y = y_base + math.sin(i * 0.2) * 0.8  # Slight breathing
            draw_human(frame, cur_x, cur_y, scale=scale, pose="standing", leg_phase=i * 0.1)

        elif activity == "Lying_Down":
            cur_x = WIDTH // 2
            cur_y = HEIGHT - 22 + math.sin(i * 0.1) * 0.5
            draw_human(frame, cur_x, cur_y, scale=scale, pose="lying_down")

        elif activity == "Falling":
            cur_x = WIDTH // 2
            if i < 7:
                # Upright initially
                cur_y = y_base
                draw_human(frame, cur_x, cur_y, scale=scale, pose="standing")
            elif i < 18:
                # Sudden slip and rotational collapse downward
                slip_t = (i - 7) / 11.0
                angle = slip_t * 85.0  # tilts up to 85 degrees
                cur_y = y_base + (slip_t ** 2) * 22  # accelerating downward
                draw_human(frame, cur_x + slip_t * 12, cur_y, scale=scale, pose="falling", angle=angle)
            else:
                # Crash impact on floor
                draw_human(frame, cur_x + 14, HEIGHT - 20, scale=scale, pose="lying_down")

        out.write(frame)

    out.release()


def generate_full_dataset(dataset_root="dataset", sample_data_dir="sample_data"):
    """
    Populates train, validation, test, and sample_data directories with realistic clips.
    """
    splits = {
        "train": 10,       # 10 samples per class
        "validation": 3,   # 3 samples per class
        "test": 3          # 3 samples per class
    }

    print("Generating synthetic Human Activity Recognition dataset...")
    total_generated = 0

    # IMPORTANT: each split and class must use a different seed. The previous
    # seed formula reused the same seed for train/validation/test, causing exact
    # byte-identical clips to leak across splits and inflating evaluation scores.
    split_offsets = {"train": 0, "validation": 100_000, "test": 200_000}
    for split, count in splits.items():
        for class_index, cls in enumerate(CLASSES):
            folder = os.path.join(dataset_root, split, cls)
            os.makedirs(folder, exist_ok=True)
            for s in range(count):
                file_name = f"{cls.lower()}_{split}_{s+1:02d}.mp4"
                file_path = os.path.join(folder, file_name)
                seed = 1000 + split_offsets[split] + class_index * 1000 + s * 17 + len(cls)
                generate_video_clip(cls, file_path, seed=seed)
                total_generated += 1

    # Also generate ready-to-test sample videos for dashboard upload
    os.makedirs(sample_data_dir, exist_ok=True)
    for cls in CLASSES:
        demo_path = os.path.join(sample_data_dir, f"demo_{cls.lower()}.mp4")
        generate_video_clip(cls, demo_path, seed=2026 + len(cls) * 31)
        print(f"Created sample demo video: {demo_path}")

    print(f"Successfully generated {total_generated} training/validation/test clips and {len(CLASSES)} demo videos.")


if __name__ == "__main__":
    generate_full_dataset()
