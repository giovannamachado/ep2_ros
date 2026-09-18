import cv2
import cv2.aruco as aruco
import numpy as np


def create_detector():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    parameters = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, parameters)

    return detector


def detect_markers(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    marker_to_slot = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8
    }

    markers = {}

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):

            marker_id = int(marker_id)

            marker_corners = corners[i][0]

            center_x = np.mean(marker_corners[:, 0])
            center_y = np.mean(marker_corners[:, 1])

            center = (int(center_x), int(center_y))

            slot = marker_to_slot.get(marker_id)

            markers[marker_id] = {
                "corners": marker_corners,
                "position": slot,
                "center": center
            }

    return markers


def main():
    camera = cv2.VideoCapture(0)

    detector = create_detector()

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Erro ao capturar imagem da câmera.")
            break

        markers = detect_markers(detector, frame)

        for marker_id, info in markers.items():

            center = info["center"]

            corners = info["corners"]

            x_min = int(np.min(corners[:, 0]))
            x_max = int(np.max(corners[:, 0]))

            y_min = int(np.min(corners[:, 1]))
            y_max = int(np.max(corners[:, 1]))

            print(
                f"ID: {marker_id} | "
                f"Posicao: {info['position']} | "
                f"Centro: {center}"
            )

            print(
                f"Limites: X({x_min}, {x_max}) | "
                f"Y({y_min}, {y_max})"
            )

            cv2.circle(
                frame,
                center,
                6,
                (0, 255, 0),
                -1
            )

        cv2.imshow("ArUco", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()