import depthai as dai
import cv2
import numpy as np
import os
import time

import st7735
from PIL import Image

from depthai_nodes.node import SnapsUploader
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from utils.snaps_producer import SnapsProducer
from dotenv import load_dotenv

# Load environment variables before initializing the pipeline.
load_dotenv(override=True)

model = "luxonis/yolov6-nano:r2-coco-512x288"
time_interval = 10.0  # min nr of seconds between snaps uploading

# --- Enviro+ LCD configuration ---
# The Enviro+ screen is a 160x80 ST7735 SPI display, driven by the separate
# `st7735` library (a dependency of enviroplus-python, not part of it).
LCD_UPDATE_INTERVAL_S = 0.5  # throttle SPI writes — no need to push every frame

# --- ArUco configuration ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
MARKER_LENGTH_M = 0.200  # 200mm x 200mm markers

ARUCO_RES = (1920, 1080)
CAMERA_SOCKET = dai.CameraBoardSocket.CAM_A  # OAK-D Lite's RGB camera

# Marker object points (centered at origin, Z=0), used with solvePnP.
# Order matches cv2.aruco corner order: top-left, top-right, bottom-right, bottom-left.
_half = MARKER_LENGTH_M / 2
OBJ_POINTS = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)


class ArucoDetectorNode(dai.node.HostNode):
    """Runs ArUco detection + pose estimation on the full-res frame and
    publishes an annotated frame as a pipeline output for the browser
    visualizer, and also pushes a throttled copy to the Enviro+'s onboard
    160x80 LCD — no local monitor required either way."""

    def build(self, frame_output, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        self.output = self.createOutput()

        # Enviro+ LCD setup (same init pattern as the library's own examples).
        self.lcd = st7735.ST7735(
            port=0,
            cs=1,
            dc="GPIO9",
            backlight="GPIO12",
            rotation=270,
            spi_speed_hz=10000000,
        )
        self.lcd.begin()
        self.lcd_width = self.lcd.width
        self.lcd_height = self.lcd.height
        self._last_lcd_update = 0.0

        self.link_args(frame_output)
        return self

    def process(self, img_frame):
        frame = img_frame.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        detections = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                ok, rvec, tvec = cv2.solvePnP(
                    OBJ_POINTS, marker_corners[0], self.camera_matrix, self.dist_coeffs
                )
                if ok:
                    cv2.drawFrameAxes(
                        frame, self.camera_matrix, self.dist_coeffs, rvec, tvec,
                        MARKER_LENGTH_M * 0.5
                    )
                    distance_m = float(np.linalg.norm(tvec))
                    x, y, z = tvec.flatten()
                    rot_deg = np.degrees(rvec.flatten())
                    center_px = marker_corners[0].mean(axis=0)
                    detections.append((int(marker_id), distance_m))

                    # Printed straight away, independent of the visualizer —
                    # useful since the browser view can lag behind real time.
                    print(
                        f"[ArUco] id={marker_id} "
                        f"dist={distance_m:.3f}m "
                        f"pos(x,y,z)=({x:.3f}, {y:.3f}, {z:.3f})m "
                        f"rot(x,y,z)=({rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f})deg "
                        f"pixel_center=({center_px[0]:.0f}, {center_px[1]:.0f})",
                        flush=True,
                    )

        out_frame = dai.ImgFrame()
        out_frame.setCvFrame(frame, dai.ImgFrame.Type.BGR888p)
        try:
            self.output.send(out_frame)
        except Exception:

            # closed out from under us — safe to just drop this last frame.
            pass

        self._update_lcd(frame)

    def _update_lcd(self, frame):
        now = time.monotonic()
        if now - self._last_lcd_update < LCD_UPDATE_INTERVAL_S:
            return
        self._last_lcd_update = now

        #scale down frame to lcd size and convert to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lcd_frame = cv2.resize(rgb_frame, (self.lcd_width, self.lcd_height))
        lcd_image = Image.fromarray(lcd_frame)
        try:
            self.lcd.display(lcd_image)
        except Exception as e:
            print(f"[LCD] failed to update display: {e}", flush=True)


visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription(model)
    platform = device.getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            model_description,
        )
    )

    input_node = pipeline.create(dai.node.Camera).build()

    
    aruco_output = input_node.requestOutput(ARUCO_RES, type=dai.ImgFrame.Type.NV12)

    # Camera intrinsics/distortion at the ArUco tap's resolution, needed for
    # solvePnP-based pose estimation.
    calib = device.readCalibration()
    camera_matrix = np.array(
        calib.getCameraIntrinsics(CAMERA_SOCKET, ARUCO_RES[0], ARUCO_RES[1])
    )
    dist_coeffs = np.array(calib.getDistortionCoefficients(CAMERA_SOCKET))

    aruco_node = pipeline.create(ArucoDetectorNode).build(
        aruco_output, camera_matrix, dist_coeffs
    )

    
    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive
    )

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")
    visualizer.addTopic("ArUco", aruco_node.output, "images")

    snaps_producer = pipeline.create(SnapsProducer).build(
        frame=nn_with_parser.passthrough,
        detections=nn_with_parser.out,
        time_interval=time_interval
    )

    snaps_uploader = pipeline.create(SnapsUploader).build(snaps_producer.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break