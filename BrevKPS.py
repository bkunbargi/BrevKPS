import numpy as np
import tempfile
import torch
import cv2
from PIL import Image
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KPSScaleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "position_x": ("FLOAT", {
                    "default": None,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                }),
                "position_y": ("FLOAT", {
                    "default": None,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_kps"
    CATEGORY = "BrevKPS"

    def scale_keypoints(self, normalized_keypoints, scale_factor, position_x=None, position_y=None):
            logger.info(f"Scaling keypoints with factor {scale_factor}")
            logger.info(f"Input keypoints: {normalized_keypoints}")
            logger.info(f"Position: ({position_x}, {position_y})")

            if scale_factor == 1.0 and position_x is None and position_y is None:
                logger.info("No changes needed - returning original keypoints")
                return normalized_keypoints
                
            nose_kp = next(kp for kp in normalized_keypoints['keypoints'] if kp['feature'] == 'Keypoint 3')
            logger.info(f"Nose keypoint: {nose_kp}")
            
            new_position = None
            if position_x is not None and position_y is not None:
                new_position = {"x": position_x, "y": position_y}

            scaled_kps = []
            for kp in normalized_keypoints['keypoints']:
                dx = (kp['x'] - nose_kp['x']) * scale_factor
                dy = (kp['y'] - nose_kp['y']) * scale_factor
                
                scaled_kp = {
                    # 'x': kp['x'] + (dx - (kp['x'] - nose_kp['x'])),
                    # 'y': kp['y'] + (dy - (kp['y'] - nose_kp['y'])),
                    'x': new_position['x'] + dx if new_position is not None else kp['x'] + dx,
                    'y': new_position['y'] + dy if new_position is not None else kp['y'] + dy,
                    'feature': kp['feature']
                }
                scaled_kps.append(scaled_kp)
                logger.info(f"Scaled {kp['feature']}: original=({kp['x']:.3f}, {kp['y']:.3f}), scaled=({scaled_kp['x']:.3f}, {scaled_kp['y']:.3f})")

            scaled_keypoints = {
                "keypoints": scaled_kps,
            }
            logger.info(f"Final scaled keypoints: {scaled_keypoints}")
            return scaled_keypoints

    def draw_keypoints_new(self, dimensions, keypoints, color_list=[(255, 0, 0), (0, 255, 0), 
                                                            (0, 0, 255), (255, 255, 0), 
                                                            (255, 0, 255)]):
        logger.info("Starting keypoint drawing")
        logger.info(f"Drawing dimensions: {dimensions}")
        logger.info(f"Input keypoints: {keypoints}")
        
        stickwidth = 4
        w, h = dimensions
        kps = np.array([[kp['x'], kp['y']] for kp in keypoints['keypoints']])
        logger.info(f"Keypoints array shape: {kps.shape}")
        
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kp_order = [0, 1, 2, 3, 4]

        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Validate number of keypoints
        if len(keypoints['keypoints']) != 5:
            logger.error(f"Expected 5 keypoints, but got {len(keypoints['keypoints'])}")
            return out_img

        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[kp_order[index[0]]]
            x = kps[index][:, 0] * w
            y = kps[index][:, 1] * h
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            
            logger.info(f"Drawing limb {i}: index={index}, length={length:.2f}, angle={angle:.2f}")
            
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
            logger.info(f"Drew keypoint {kp_idx + 1} at ({x}, {y})")
        
        logger.info("Finished drawing keypoints")
        return out_img

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

    def _decode_image_from_bytes(self, image_bytes):
        """Decode image from bytes using OpenCV."""
        image_np = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    def process_kps(self, image, scale_factor, position_x=None, position_y=None):
        try:
            logger.info("Starting KPS processing")
            logger.info(f"Scale factor: {scale_factor}")
            if isinstance(image, torch.Tensor):
                logger.info("IMAGE IS TENSOR")
                image_np = image.squeeze(0).permute(0,1,2).cpu().numpy()
            else:
                logger.info("IMAGE IS PIL")
                image_np = np.array(image)

            if image_np.shape[2] != 3:
                image_np = np.transpose(image_np, (1,2,0))

            image_rgb = (image_np * 255).clip(0, 255).astype(np.uint8)
            
            h, w = image_rgb.shape[:2]
            logger.info(f"Image dimensions: {w}x{h}")
            
            # Convert numpy array to PIL Image for saving
            pil_image = Image.fromarray(image_rgb)
            
            # Use temporary file to save and read the image
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
                logger.info(f"Saving temporary file to: {temp_file.name}")
                pil_image.save(temp_file.name, format='PNG')
                # Read the file in binary mode and decode
                image_bytes = open(temp_file.name, 'rb').read()
                decoded_image = self._decode_image_from_bytes(image_bytes)
                # Now call get_coords with the decoded image
                keypoints = self.get_coords(decoded_image)
            
            # Scale keypoints
            scaled_keypoints = self.scale_keypoints(keypoints, scale_factor, position_x, position_y)
            
            # Draw the new keypoint image
            output_img = self.draw_keypoints_new((w, h), scaled_keypoints)
            
            output_array = np.array(output_img).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_array).unsqueeze(0)
            
            logger.info("KPS processing completed successfully")
            return (output_tensor,)
        except Exception as e:
            logger.error(f"Error in process_kps: {str(e)}")
            raise

# Update the node mappings
NODE_CLASS_MAPPINGS = {
    "KPSScaleNode": KPSScaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KPSScaleNode": "KPS Scale"
}