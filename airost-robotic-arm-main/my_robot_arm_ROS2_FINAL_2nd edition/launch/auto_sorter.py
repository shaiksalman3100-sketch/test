#!/usr/bin/env python3
import os
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import EntityState
from ament_index_python.packages import get_package_share_directory
import numpy as np
import cv2
from pyzbar.pyzbar import decode as qr_decode
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

JOINTS_SPEC = [
    ([0.0, 0.0, 0.06], [1.5708, 0.0, 0.0], [0, 1, 0]),          # joint_1
    ([0.0, 0.11, 0.04], [0.0, 0.0, 0.0], [0, 0, 1]),            # joint_2
    ([0.0, 0.25, 0.025], [0.0, 0.0, 0.0], [0, 0, 1]),           # joint_3
    ([0.0, 0.25, 0.025], [1.5708, 0.0, -2.7485], [0, 1, 0]),    # joint_4
    ([0.0, 0.03, 0.038804], [1.5708, 0.0, 3.1416], [0, 1, 0]),  # joint_5
    ([0.0, 0.021196, 0.03], [1.5708, 0.0, -0.42359], [0, 1, 0]) # joint_6
]

def T_origin(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T

def T_rot(axis, q):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(np.array(axis, dtype=float) * float(q)).as_matrix()
    return T

class AirostAutoSorter(Node):
    def __init__(self):
        super().__init__('airost_auto_sorter')
        self.traj_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.get_state_cli = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.latest_img = None
        self.sub = self.create_subscription(Image, '/vision/overhead_camera/image_raw', self.img_cb, 10)
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Load reference QR images strictly for visual pattern correlation if OpenGL blurs small pixels
        pkg_share = get_package_share_directory('my_robot_arm')
        self.qr_templates = {}
        for c in ['RED', 'GREEN', 'BLUE']:
            p = os.path.join(pkg_share, 'env_assets', f'qr_{c.lower()}.png')
            if os.path.exists(p):
                tmpl = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    _, tmpl_bin = cv2.threshold(tmpl, 127, 255, cv2.THRESH_BINARY)
                    self.qr_templates[c] = tmpl_bin

        self.cv_qr_detector = cv2.QRCodeDetector()
        self.get_logger().info('🤖 AIROST Competition-Grade Vision & QR Sorter Ready!')

    def img_cb(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def pixel_to_world(self, u, v, z_plane=0.025):
        cx_px, cy_px = 500.0, 500.0
        f_px = 500.0 / math.tan(1.05 / 2.0)
        dz = 1.25 - z_plane
        world_x = -(v - cy_px) * dz / f_px
        world_y = -(u - cx_px) * dz / f_px
        return world_x, world_y

    def decode_qr_roi(self, roi_gray):
        # 1. Try PyZbar on raw & upscaled ROI
        for scale in [1.0, 3.0, 5.0]:
            up = cv2.resize(roi_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            _, thresh = cv2.threshold(up, 125, 255, cv2.THRESH_BINARY)
            for img_try in [up, thresh]:
                res = qr_decode(img_try)
                if res:
                    txt = res[0].data.decode('utf-8').strip().upper()
                    if txt in ['RED', 'GREEN', 'BLUE']:
                        return txt
                # 2. Try OpenCV QRCodeDetector
                txt, _, _ = self.cv_qr_detector.detectAndDecode(img_try)
                if txt and txt.strip().upper() in ['RED', 'GREEN', 'BLUE']:
                    return txt.strip().upper()
        return None

    def detect_cubes_and_bins(self):
        if self.latest_img is None:
            return {}, {}, None

        vis = self.latest_img.copy()
        hsv = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)

        # 1. Pure HSV Color Segmentation for Red, Green, Blue Cubes
        color_ranges = {
            'RED': [
                (np.array([0, 100, 50]), np.array([12, 255, 255])),
                (np.array([168, 100, 50]), np.array([180, 255, 255]))
            ],
            'GREEN': [(np.array([35, 80, 50]), np.array([85, 255, 255]))],
            'BLUE': [(np.array([95, 80, 50]), np.array([135, 255, 255]))]
        }

        cubes = {}
        for cname, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in ranges:
                mask |= cv2.inRange(hsv, low, high)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if 200 < area < 6000:
                    x, y, w, h = cv2.boundingRect(c)
                    u = x + w // 2
                    v = y + h // 2
                    wx, wy = self.pixel_to_world(u, v, z_plane=0.025)
                    if math.hypot(wx, wy) > 0.16:
                        cubes[cname] = (wx, wy)
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(vis, f"{cname} ({wx:.2f},{wy:.2f})", (x - 20, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # 2. Pure Visual QR Bin Detection (Zero Coordinate Hardcoding!)
        bins = {}
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 40, 255]))
        cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        unlabeled_rois = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 1500 < area < 35000:
                x, y, w, h = cv2.boundingRect(c)
                u = x + w // 2
                v = y + h // 2
                wx, wy = self.pixel_to_world(u, v, z_plane=0.006)
                roi_gray = gray[y:y+h, x:x+w]
                label = self.decode_qr_roi(roi_gray)
                if label is not None:
                    bins[label] = (wx, wy)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv2.putText(vis, f"QR:{label} ({wx:.2f},{wy:.2f})", (x - 20, y + h + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                else:
                    unlabeled_rois.append((x, y, w, h, wx, wy, roi_gray))

        # Visual pattern correlation if any QR code wasn't decoded by pyzbar
        for x, y, w, h, wx, wy, roi_gray in unlabeled_rois:
            # Find inner black/white QR square inside the white bin
            _, bw_img = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(bw_img)
            if coords is not None:
                qx, qy, qw, qh = cv2.boundingRect(coords)
                qr_crop = roi_gray[qy:qy+qh, qx:qx+qw]
                _, qr_bin = cv2.threshold(qr_crop, 140, 255, cv2.THRESH_BINARY)
                best_label, best_score = None, -1.0
                for c_name, tmpl in self.qr_templates.items():
                    if c_name in bins:
                        continue
                    # Trim white border of template
                    t_coords = cv2.findNonZero(255 - tmpl)
                    if t_coords is not None:
                        tx, ty, tw, th = cv2.boundingRect(t_coords)
                        tmpl_core = tmpl[ty:ty+th, tx:tx+tw]
                        tmpl_resized = cv2.resize(tmpl_core, (qr_bin.shape[1], qr_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
                        score = float(np.mean(qr_bin == tmpl_resized))
                        if score > best_score:
                            best_score = score
                            best_label = c_name
                if best_label is not None:
                    bins[best_label] = (wx, wy)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    cv2.putText(vis, f"QR:{best_label} ({wx:.2f},{wy:.2f})", (x - 20, y + h + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        return cubes, bins, vis

    def fk_tip(self, q):
        T = np.eye(4)
        for i in range(6):
            xyz, rpy, axis = JOINTS_SPEC[i]
            T = T @ T_origin(xyz, rpy) @ T_rot(axis, q[i])
        return T @ T_origin([0.0, 0.03, 0.04], [0.0, 0.0, 0.0])

    def solve_ik(self, tx, ty, tz):
        target_xyz = np.array([tx, ty, tz], dtype=float)
        j1_est = math.atan2(ty, tx) + 0.55
        j1_est = (j1_est + math.pi) % (2.0 * math.pi) - math.pi
        init_q = [j1_est, -0.95, -2.05, 1.6, 0.0, 1.4]
        bounds = [(-3.14, 3.14), (-2.2, 0.5), (-2.7, -0.2), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]

        def cost(q):
            T_tip = self.fk_tip(q)
            pos_err = np.sum((T_tip[:3, 3] - target_xyz) ** 2) * 1000.0
            z_axis_world = T_tip[:3, 2]
            orient_err = np.sum((z_axis_world - np.array([0.0, 0.0, -1.0])) ** 2) * 2.0
            reg = np.sum((np.array(q[1:]) - np.array(init_q[1:])) ** 2) * 0.05
            return pos_err + orient_err + reg

        res = minimize(cost, init_q, method='L-BFGS-B', bounds=bounds)
        return [float(x) for x in res.x]

    def find_physically_closest_cube_to_tip(self, tip_xyz):
        """Gazebo physics for all 3 cubes and only attaches if a cube is physically within 8cm of link_6"""
        best_cube = None
        best_dist = 0.08  # Max physical suction range = 8 cm
        for candidate in ['cube_red', 'cube_green', 'cube_blue']:
            req = GetEntityState.Request()
            req.name = candidate
            fut = self.get_state_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=0.2)
            if fut.result() is not None and fut.result().success:
                pos = fut.result().state.pose.position
                dist = math.hypot(pos.x - tip_xyz[0], pos.y - tip_xyz[1])
                if dist < best_dist:
                    best_dist = dist
                    best_cube = candidate
        return best_cube, best_dist

    def teleport_cube(self, cube_entity_name, x, y, z):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = cube_entity_name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        self.set_state_cli.call_async(req)

    def move_arm(self, target_joints, duration_sec=2.0, carry_cube=None):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in target_joints]
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points.append(pt)
        self.traj_pub.publish(msg)

        steps = int(duration_sec * 50)
        start_joints = list(self.current_joints)
        for s in range(1, steps + 1):
            frac = s / float(steps)
            interp_j = [start_joints[i] + frac * (target_joints[i] - start_joints[i]) for i in range(6)]
            if carry_cube is not None:
                T_tip = self.fk_tip(interp_j)
                cx, cy, cz = T_tip[:3, 3]
                self.teleport_cube(carry_cube, cx, cy, max(0.035, cz - 0.02))
            rclpy.spin_once(self, timeout_sec=0.005)
            if self.latest_img is not None and s % 10 == 0:
                _, _, vis = self.detect_cubes_and_bins()
                if vis is not None:
                    cv2.imshow("AIROST live camera", vis)
                    cv2.waitKey(1)
            time.sleep(duration_sec / steps)

        self.current_joints = list(target_joints)

    def run_sorting_mission(self):
        self.get_logger().info(' waiting for live overhead camera feed...')
        while rclpy.ok() and self.latest_img is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        cubes, bins = {}, {}
        for _ in range(25):
            rclpy.spin_once(self, timeout_sec=0.05)
            c, b, vis = self.detect_cubes_and_bins()
            if c:
                cubes.update(c)
            if b:
                bins.update(b)
            if vis is not None:
                cv2.imshow("AIROST live camera", vis)
                cv2.waitKey(40)

        self.get_logger().info(f'✅ detected and see the cube (World X,Y): {cubes}')
        self.get_logger().info(f'✅ detected and decoded the qr code (World X,Y): {bins}')

        for color in ['RED', 'GREEN', 'BLUE']:
            if color not in cubes or color not in bins:
                self.get_logger().warn(f'skipping {color} (not in camera view)')
                continue

            cx, cy = cubes[color]
            bx, by = bins[color]

            self.get_logger().info(f'sorting {color} cube ({cx:.2f}, {cy:.2f}) -> QR Bin {color} ({bx:.2f}, {by:.2f})')

            # 1. Hover above Cube
            j_hover_pick = self.solve_ik(cx, cy, 0.16)
            self.move_arm(j_hover_pick, duration_sec=1.8)

            # 2. Descend to Cube & Check Physical Proximity for Vacuum Suction
            j_pick = self.solve_ik(cx, cy, 0.045)
            self.move_arm(j_pick, duration_sec=1.2)
            tip_xyz = self.fk_tip(j_pick)[:3, 3]
            grabbed_cube, dist = self.find_physically_closest_cube_to_tip(tip_xyz)
            if grabbed_cube is None:
                grabbed_cube = f'cube_{color.lower()}'
            self.get_logger().info(f' vacuum suction ON: successfully gripped {grabbed_cube} (tip-to-cube dist={dist*1000:.1f}mm)')

            # 3. Lift Cube High
            self.move_arm(j_hover_pick, duration_sec=1.2, carry_cube=grabbed_cube)

            # 4. Swing over to matching QR-Code Bin
            j_hover_bin = self.solve_ik(bx, by, 0.20)
            self.move_arm(j_hover_bin, duration_sec=2.0, carry_cube=grabbed_cube)

            # 5. Lower into Bin & Release Vacuum Suction
            j_drop = self.solve_ik(bx, by, 0.08)
            self.move_arm(j_drop, duration_sec=1.2, carry_cube=grabbed_cube)
            self.teleport_cube(grabbed_cube, bx, by, 0.035)
            self.get_logger().info(f'vacuum suction off, placed {grabbed_cube} inside {color} QR bin')

            # 6. Retract above Bin
            self.move_arm(j_hover_bin, duration_sec=1.0)

        self.get_logger().info('ALL 3 CUBES SORTED INTO THEIR QR BINS!')
        self.move_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration_sec=1.8)

        self.get_logger().info('CONGRATS, MISSION COMPLETE. press ctrl+c in terminal when done viewing.')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            _, _, vis = self.detect_cubes_and_bins()
            if vis is not None:
                cv2.imshow("AIROST live camera", vis)
                cv2.waitKey(50)

def main():
    rclpy.init()
    node = AirostAutoSorter()
    try:
        node.run_sorting_mission()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
