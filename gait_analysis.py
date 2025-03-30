import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from utils.gait_utils import get_ankle_coord, calculate_step_variation
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
os.makedirs("results", exist_ok=True)
os.makedirs("results/frames", exist_ok=True)

cap = cv2.VideoCapture("videos/walking.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

debug_writer = cv2.VideoWriter(
    "results/output_with_debug.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
summary_writer = cv2.VideoWriter(
    "results/gait_result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

step_lengths = []
frame_idx = 0
prev_side = None
step_num = 0
last_frame = None
movement_axis = "x"
MIN_STEP_THRESHOLD = 0.015
STANCE_Y_THRESHOLD = 0.03

print("Starting gait analysis...")
print("-" * 40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    last_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=DrawingSpec(color=(0, 0, 255), thickness=2),
        )

        l_x = get_ankle_coord(landmarks, "left", axis="x")
        r_x = get_ankle_coord(landmarks, "right", axis="x")
        l_y = get_ankle_coord(landmarks, "left", axis="y")
        r_y = get_ankle_coord(landmarks, "right", axis="y")

        if None in [l_x, r_x, l_y, r_y]:
            print(f"[Frame {frame_idx}] 🚫 One or more ankle landmarks not detected")
        else:
            step_diff = abs(l_x - r_x)
            current_side = "left" if l_x < r_x else "right"
            y_diff = abs(l_y - r_y)
            in_stance_phase = y_diff < STANCE_Y_THRESHOLD

            print(
                f"[Frame {frame_idx}] X Diff: {round(step_diff, 4)} | Y Diff: {round(y_diff, 4)} | Side: {current_side} | In Stance: {in_stance_phase}"
            )

            if in_stance_phase:
                l_coords = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_coords = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                cv2.circle(
                    frame,
                    (int(l_coords.x * width), int(l_coords.y * height)),
                    8,
                    (0, 255, 0),
                    -1,
                )
                cv2.circle(
                    frame,
                    (int(r_coords.x * width), int(r_coords.y * height)),
                    8,
                    (0, 255, 0),
                    -1,
                )

            if (
                prev_side
                and current_side != prev_side
                and step_diff > MIN_STEP_THRESHOLD
                and in_stance_phase
            ):
                step_num += 1
                step_lengths.append(step_diff)

                print(f"✅ Step {step_num} at frame {frame_idx}")
                print(f"   → Side switched to {current_side}")
                print(f"   → X diff: {round(step_diff, 4)}\n")

                cv2.putText(
                    frame,
                    f"Step {step_num}: {current_side.upper()}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 200, 50),
                    2,
                )
                cv2.putText(
                    frame,
                    f"X Diff: {round(step_diff, 4)}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            prev_side = current_side

    else:
        print(f"[Frame {frame_idx}] Pose not detected")

    debug_writer.write(frame)
    cv2.imwrite(f"results/frames/frame_{frame_idx:04d}.png", frame)

    cv2.imshow("Frame Debug", frame)
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

    frame_idx += 1

cap.release()
debug_writer.release()
cv2.destroyAllWindows()

if len(step_lengths) < 2:
    print("Not enough steps detected.")
else:
    analysis = calculate_step_variation(step_lengths)

    print("\nGait Analysis Results")
    print("------------------------")
    print("Step Lengths:", [round(s, 4) for s in step_lengths])
    print("Mean Step Length:", round(analysis["mean"], 4))
    print("Standard Deviation:", round(analysis["std_dev"], 4))
    print("Coefficient of Variation:", round(analysis["coefficient_of_variation"], 4))
    print("Result:", "EVEN steps" if analysis["even_steps"] else "UNEVEN steps")

    result_text = "EVEN steps" if analysis["even_steps"] else "UNEVEN steps"
    color = (0, 255, 0) if analysis["even_steps"] else (0, 0, 255)

    cv2.putText(
        last_frame,
        f"Result: {result_text}",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
    )
    cv2.putText(
        last_frame,
        f"Coeff of Var: {round(analysis['coefficient_of_variation'], 4)}",
        (50, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    for _ in range(int(fps * 3)):
        summary_writer.write(last_frame)

    summary_writer.release()
    print("Saved debug video and result summary in 'results/' folder.")
