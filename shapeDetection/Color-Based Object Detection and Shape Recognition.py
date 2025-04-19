import cv2
import numpy as np

def nothing(x):
    pass

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Trackbar Penceresi
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)   # Alt Hue
cv2.createTrackbar("L-S", "Trackbars", 97, 255, nothing)  # Alt Saturation
cv2.createTrackbar("L-V", "Trackbars", 127, 255, nothing) # Alt Value
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing) # Üst Hue
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing) # Üst Saturation
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing) # Üst Value

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kamera görüntüsü alinamadi!")
        break

    # BGR'den HSV'ye dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbar'dan değerleri oku
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    # HSV Alt ve Üst sınırları belirle
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Maske oluştur
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Gürültü azaltma
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Konturları kontrol et ve çiz
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 4)  # Siyah çizgi ile kontur çiz

            # Moment hesaplama (Merkez noktasını bulma)
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # Bölme hatasını önlemek için kontrol
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Merkeze yeşil bir nokta koy
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Koordinatları ekrana yazdır
                cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy - 10), font, 0.5, (255, 255, 255), 1)

            # Şekil belirleme
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if len(approx) == 4:
                cv2.putText(frame, "Dikdortgen", (x, y), font, 1, (0, 0, 0))
            elif len(approx) == 3:
                cv2.putText(frame, "Ucgen", (x, y), font, 1, (0, 0, 0))
            elif 10 < len(approx) < 20:
                cv2.putText(frame, "Daire", (x, y), font, 1, (0, 0, 0))

    # Görüntüleri göster
    cv2.imshow("Frame", frame)  # Orijinal görüntü
    cv2.imshow("Mask", mask)    # Maske görüntüsü

    # Çıkış için 'Esc' tuşuna bas
    key = cv2.waitKey(1)
    if key == 27:  # ESC tuşunun ASCII kodu 27
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
