import cv2

static_back = None
cap = cv2.VideoCapture(0)
cap.set(10, 100)

while True:
    ret, frame = cap.read()
    Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    Blur = cv2.GaussianBlur(Gray, (21, 21), 0)
    if static_back is None:
        static_back = Gray
        continue

    diff_frame = cv2.absdiff(static_back, Gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        t = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(t, 'Motion Detected', (x, y), font, 0.5, (0, 0, 255), 2)

        cv2.imshow("contour", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

