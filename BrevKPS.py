import numpy as np
import torch
import cv2
from PIL import Image
import math

class KPSScalePositionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "change_position": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "position_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "position_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_kps"
    CATEGORY = "BrevKPS"

    def get_coords(self, img):
        centers = []
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        for target_color in colors:
            mask = cv2.inRange(img, np.array(target_color), np.array(target_color))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) / w
                    cY = int(M["m01"] / M["m00"]) / h
                    centers.append((cX, cY))
        keypoints_formatted = {'keypoints': [{'x': x, 'y': y, 'feature': f"Keypoint {i}"} for i, (x, y) in enumerate(centers, start=1)]}
        return keypoints_formatted

    def scale_and_position_keypoints(self, normalized_keypoints, scale_factor, new_position=None):
        print("Scale and Position: ", normalized_keypoints)
        print("Scale factor:", scale_factor)
        print("New position:", new_position)
        
        nose_kp = next(kp for kp in normalized_keypoints['keypoints'] if kp['feature'] == 'Keypoint 3')
        scaled_kps = []
        
        for kp in normalized_keypoints['keypoints']:
            dx = (kp['x'] - nose_kp['x']) * scale_factor
            dy = (kp['y'] - nose_kp['y']) * scale_factor
            
            if new_position is not None:
                x = new_position['x'] + dx
                y = new_position['y'] + dy
            else:
                # Keep the nose point at its original position and scale around it
                x = nose_kp['x'] + dx
                y = nose_kp['y'] + dy
                
            scaled_kp = {
                'x': x,
                'y': y,
                'feature': kp['feature']
            }
            scaled_kps.append(scaled_kp)

        scaled_keypoints = {
            "keypoints": scaled_kps,
        }
        return scaled_keypoints

    def draw_keypoints_new(self, dimensions, keypoints, color_list=[(255, 0, 0), (0, 255, 0), 
                                                            (0, 0, 255), (255, 255, 0), 
                                                            (255, 0, 255)]):
        stickwidth = 4
        w, h = dimensions
        kps = np.array([[kp['x'], kp['y']] for kp in keypoints['keypoints']])
        
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kp_order = [0, 1, 2, 3, 4]

        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[kp_order[index[0]]]
            x = kps[index][:, 0] * w
            y = kps[index][:, 1] * h
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), 
                                       (int(length / 2), stickwidth), 
                                       int(angle), 0, 360, 1)
            out_img = cv2.fillConvexPoly(out_img, polygon, color)

        out_img = (out_img * 0.6).astype(np.uint8)

        for idx, kp_idx in enumerate(kp_order):
            color = color_list[idx]
            kp = keypoints['keypoints'][kp_idx]
            x, y = int(kp['x'] * w), int(kp['y'] * h) 
            out_img = cv2.circle(out_img, (x, y), 10, color, -1)
        
        return out_img

    def process_kps(self, image, scale_factor, change_position, position_x=None, position_y=None):
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.squeeze(0).permute(0,1,2).cpu().numpy()
            else:
                image_np = np.array(image)

            if image_np.shape[2] != 3:
                image_np = np.transpose(image_np, (1,2,0))

            image_rgb = (image_np * 255).clip(0, 255).astype(np.uint8)
            
            # Get original dimensions
            h, w = image_rgb.shape[:2]
            
            # Extract keypoints from input image
            keypoints = self.get_coords(image_rgb)
            
            # Only use position if change_position is True
            new_position = {'x': position_x, 'y': position_y} if change_position else None
            scaled_keypoints = self.scale_and_position_keypoints(keypoints, scale_factor, new_position)
            
            # Draw the new keypoint image
            output_img = self.draw_keypoints_new((w, h), scaled_keypoints)
            
            # Convert back to ComfyUI image format
            output_array = np.array(output_img).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_array).unsqueeze(0)
            
            return (output_tensor,)
            
        except Exception as e:
            print(f"Error in process_kps: {str(e)}")
            import traceback
            traceback.print_exc()
            return (image,)

# Update the node mappings
NODE_CLASS_MAPPINGS = {
    "KPSScalePositionNode": KPSScalePositionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KPSScalePositionNode": "KPS Scale and Position"
}