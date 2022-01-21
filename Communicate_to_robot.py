import sys
import socket
import select
import time

import cv2
from os.path import join, dirname
from src.calibrator import Calibrator
from src.camera import Camera
from src.detector import Detector

HOST = '192.168.0.6'
PORT = 50000
PROJECT_PATH = dirname(__file__)
MODEL_FOLDER = join(PROJECT_PATH, "models", "yolov5")
MODEL_PATH = join(MODEL_FOLDER, "better_all_parts_yolov5m.pt")
CALIBRATION_DATA_PATH = join(PROJECT_PATH, "src", "calibration_data.npz")

class RobotCommand:
    def __init__(self) -> None:
        self.mode = 0
        self.part_count = 0
        self.label = 0
        self.side_pick = 0
        self.x = 0
        self.y = 0
        self.angle = 0

    def __str__(self) -> str:
        #print(self.mode, self.part_count, self.label, self.side_pick, self.x, self.y, self.angle)
        return "{:d};{:d};{};{:d};{:.2f};{:.2f};{:d}\r\n;" \
            .format(self.mode, self.part_count, self.label, self.side_pick, self.x, self.y, self.angle)

def GenerateResponse(detector : Detector, calibrator : Calibrator) -> str:
    command = RobotCommand()

    detection = detector.detect_image(camera.get_image(), True)

    if detection.label is None:
        print("Unable to get a frame from the camera")
        return None

    command.x, command.y = calibrator.project_point(detection.pick_point)
    command.part_count = detection.parts_count
    command.label = detection.label
    command.side_pick = detection.side
    command.angle = detection.angle

    if detection.parts_count == 0:
        command.mode = 9
    elif detection.unknown:
        command.mode = 2
    elif detection.do_shake:
        command.mode = 5
    else:
        command.mode = 1

    #cv2.imwrite("rapsakuva.png", detection.frame)
    cv2.imshow("frame", detection.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(1)

    return str(command)


if __name__ == "__main__":

    roi = [230, 0, 2150, 1920]
    camera = Camera(CALIBRATION_DATA_PATH, roi, init_time=125000)
    detector = Detector(640, roi)
    detector.load_model(MODEL_FOLDER, MODEL_PATH, 0.90, 0.45)
    calibrator = Calibrator(CALIBRATION_DATA_PATH)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("#########################")
        print("Starting server")
        print("#########################")

        s.bind((HOST, PORT))
        s.settimeout(1)
        s.listen()
        try:
        
            while True:

                try:
                    conn, addr = s.accept()
                except(KeyboardInterrupt):
                    print("keyboard interrupt")
                    sys.exit(1)
                except(socket.timeout):
                    continue

                with conn:
                    s.setblocking(0)
                    print('Connected by', addr)
                    #last_msg_time = time.time()
                    try:
                        while True:
                            ready = select.select([conn], [], [], 5)[0]
                            if ready:
                                data = conn.recv(32)
                                if not data:
                                    #if time.time() - last_msg_time > 30:
                                    #    print("Timeout, closing connection to robot")
                                    #    break
                                    continue
                            else:
                                #print("random else branch")
                                #if time.time() - last_msg_time > 30:
                                #    print("Timeout, closing connection to robot")
                                #    break
                                continue

                            #last_msg_time = time.time()
                            msg = data.decode("utf-8")

                            print("received,", msg[0])
                            if msg[0] == "T":

                                response = None
                                tries = 0
                                while response is None and tries < 5:
                                    response = GenerateResponse(detector, calibrator)
                                    tries += 1

                                if response is None:
                                    print("Detection failed")
                                    continue

                                data = bytes(response, 'utf-8')
                                print("Sending:", data)
                                conn.sendall(data)
                    except (KeyboardInterrupt, ConnectionAbortedError):
                        print(f"Got Error or Interrupt, closing the connection")
                        print("Waiting for new connection, close with ctrl+C")
                        pass
                    s.setblocking(1)
                    s.settimeout(3)
        except KeyboardInterrupt as interrupt:
            print("keyboard interrupt, exiting....", interrupt)
            s.close()
            sys.exit(1)