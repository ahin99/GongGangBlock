import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 얼굴 랜드마크 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 노트북 웹캠 연결 (0번 카메라)
cap = cv2.VideoCapture(0)

print(" [알림] 웹캠이 켜졌습니다. 종료하시려면 화면 창을 클릭하고 키보드의 'q'를 누르세요.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("웹캠을 불러올 수 없습니다.")
    break

  # 거울 모드처럼 보이도록 좌우 반전 및 색상 변환
  frame = cv2.flip(frame, 1)
  h, w, c = frame.shape
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # 얼굴 랜드마크 감지 수행
  results = face_mesh.process(rgb_frame)

  happiness_score = 0

  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      landmarks = face_landmarks.landmark

      # 주요 얼굴 부위 좌표 추출 (MediaPipe 인덱스 기준)
      # 입꼬리 좌우: 61번, 291번 / 윗입술 중앙: 13번
      # 얼굴 좌우 끝 (크기 정규화용): 234번, 454번
      p_left = np.array([landmarks[61].x * w, landmarks[61].y * h])
      p_right = np.array([landmarks[291].x * w, landmarks[291].y * h])
      p_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
      p_edge_l = np.array([landmarks[234].x * w, landmarks[234].y * h])
      p_edge_r = np.array([landmarks[454].x * w, landmarks[454].y * h])

      # 얼굴 전체 너비와 입꼬리 너비 계산
      face_width = np.linalg.norm(p_edge_l - p_edge_r)
      mouth_width = np.linalg.norm(p_left - p_right)

      # 입술 중앙 대비 입꼬리가 올라간 정도 계산
      mouth_center_y = (p_left[1] + p_right[1]) / 2
      lift = p_top[1] - mouth_center_y

      # 미소 비율을 바탕으로 0~100점 사이의 행복도 점수로 변환
      ratio = mouth_width / face_width
      score = (ratio - 0.25) * 450 + (lift * 0.8)
      happiness_score = int(np.clip(score, 0, 100))

  # 화면 상단에 행복도 점수 글자 띄우기
  cv2.putText(
      frame,
      f"Happiness: {happiness_score} / 100",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
  )

  # 화면 상단에 초록색 행복도 게이지(막대바) 그리기
  bar_width = int(happiness_score * 3)
  cv2.rectangle(frame, (30, 70), (30 + 300, 95), (200, 200, 200), -1)
  cv2.rectangle(frame, (30, 70), (30 + bar_width, 95), (0, 255, 0), -1)

  # 실시간 화면 출력
  cv2.imshow("Happiness Detector", frame)

  # 키보드 'q'를 누르면 프로그램 종료
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()