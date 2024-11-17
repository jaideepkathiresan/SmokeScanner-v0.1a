import cv2
import numpy as np
import glob
import os
import gc
import random  
from tqdm import tqdm
from skimage.filters import gabor
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.morphology import white_tophat, black_tophat, disk
from skimage.restoration import denoise_bilateral
from skimage.filters import threshold_local, laplace
from scipy.ndimage import gaussian_laplace
import base64
from io import BytesIO
from PIL import Image

class detectIMG:
    def __init__(self, path):
        self.img_path = path
        self.fname = os.path.basename(path)

        self.params = {
            'weight_h': 0.2,  # Reduced hue influence
            'weight_s': 0.2,  # Reduced saturation influence
            'weight_v': 0.6,  # Increased brightness influence
            'base_corr_threshold': 0.5,  # Lower threshold for more sensitive detection

            'texture_thresh': 10,  # More sensitive texture detection
            'clahe_clip_limit': 2.0,  # CLAHE to improve local contrast in smoke
            'clahe_tile_grid_size': 6,  # Tile grid size for CLAHE

            'anisotropic_iterations': 10,  # Number of iterations for anisotropic diffusion
            'anisotropic_kappa': 50,  # Sensitivity to edges for anisotropic diffusion            
            'anisotropic_gamma': 0.1,  # Rate of diffusion for anisotropic diffusion

            'tophat_selem_size': 15,  # Size of the structuring element for top-hat and bottom-hat transformations

            'log_sigma': 2,  # Sigma value for Laplacian of Gaussian

            'lbp_radius': 3,  # Radius for Local Binary Pattern
            'lbp_n_points': 24,  # Number of points for Local Binary Pattern

            'gabor_frequencies': [0.1, 0.2, 0.3],  # Frequencies for Gabor filters

            'morph_kernel_size': 7,  # Slightly larger morphological kernel
            'entropy_thresh': 80,    # Fine-tune entropy to capture smoke texture

            'weight_texture': 0.7,   # Give more weight to texture-based smoke features
            'weight_entropy': 0.3,   # Less influence from entropy filtering

            'trainedPath': "./assets/lab/"
        }

        self.color_features = [np.ones((180,), dtype=np.float32), np.ones((256,), dtype=np.float32), np.ones((256,), dtype=np.float32)]
        self.texture_feature = np.ones((256,), dtype=np.float32)
        

    # CLAHE ( for contrast enhancement )
    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.params['clahe_clip_limit'], tileGridSize=(self.params['clahe_tile_grid_size'], self.params['clahe_tile_grid_size']))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced_img

    # Anisotropic Diffusion ( for edge-preserving smoothing )
    def anisotropic_diffusion(self, img):
        iterations = self.params.get('anisotropic_iterations', 10)  # Number of iterations for anisotropic diffusion
        kappa = self.params.get('anisotropic_kappa', 50)  # Sensitivity to edges
        gamma = self.params.get('anisotropic_gamma', 0.1)  # Rate of diffusion
        img = img.astype(np.float32)
        for i in range(iterations):
            grad_n = np.roll(img, -1, axis=0) - img
            grad_s = np.roll(img, 1, axis=0) - img
            grad_e = np.roll(img, -1, axis=1) - img
            grad_w = np.roll(img, 1, axis=1) - img
            c_n = np.exp(-(grad_n / kappa) ** 2)
            c_s = np.exp(-(grad_s / kappa) ** 2)
            c_e = np.exp(-(grad_e / kappa) ** 2)
            c_w = np.exp(-(grad_w / kappa) ** 2)
            img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        return img

    # Top-hat and Bottom-hat transformations ( for small feature enhancement )
    def apply_tophat_bothat_transforms(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        selem_size = self.params.get('tophat_selem_size', 15)  # Size of the structuring element for top-hat and bottom-hat transformations
        top_hat = white_tophat(gray_img, disk(selem_size))
        bottom_hat = black_tophat(gray_img, disk(selem_size))
        return top_hat, bottom_hat

    # Laplacian of Gaussian (LoG) ( for Multi-scale edge detection )
    def multi_scale_edge_detection(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sigma = self.params.get('log_sigma', 2)  # Sigma value for Gaussian smoothing
        edge_laplacian = gaussian_laplace(gray_img, sigma=sigma)
        return edge_laplacian

    # Local Binary Patterns (LBP) ( for Texture analysis )
    def extract_lbp_features(self, img):
        radius = self.params.get('lbp_radius', 3)  # Radius of LBP
        n_points = self.params.get('lbp_n_points', 24)  # Number of points in LBP
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = np.clip(gray_img, 0, 1)  # Clip values to ensure they are between 0 and 1
        gray_img = img_as_ubyte(gray_img)  # Convert to unsigned 8-bit integer type to avoid warnings
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')


        return lbp

    # Smoke segmentation using combined features
    def scanIMG(self, lb_path):
        img = cv2.imread(self.img_path)
        lb = cv2.imread(lb_path)
        red_mask = cv2.inRange(lb, (0, 0, 128), (0, 0, 128))

        # Masking the smoke region from the original image using the red mask
        try:
            smoke_region = cv2.bitwise_and(img, img, mask=red_mask)
        except:
            return [False, None]

        # Converting to HSV and compute histograms only for the smoke region
        hsv = cv2.cvtColor(smoke_region, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], red_mask, [180], [0, 180]).astype(np.float32)
        hist_s = cv2.calcHist([hsv], [1], red_mask, [256], [0, 256]).astype(np.float32)
        hist_v = cv2.calcHist([hsv], [2], red_mask, [256], [0, 256]).astype(np.float32)

        # Comparing histograms with smoke features using correlation
        corr_h = cv2.compareHist(hist_h, self.color_features[0].astype(np.float32), cv2.HISTCMP_CORREL)
        corr_s = cv2.compareHist(hist_s, self.color_features[1].astype(np.float32), cv2.HISTCMP_CORREL)
        corr_v = cv2.compareHist(hist_v, self.color_features[2].astype(np.float32), cv2.HISTCMP_CORREL)

        # Weighted correlation
        corr_total = (self.params['weight_h'] * corr_h + self.params['weight_s'] * corr_s + self.params['weight_v'] * corr_v)

        # Thresholding based on correlation
        if corr_total < self.params['base_corr_threshold']:
            return [False, None]

        # Using Gabor filters ( for texture analysis )
        gray = cv2.cvtColor(smoke_region, cv2.COLOR_BGR2GRAY)
        frequencies = self.params.get('gabor_frequencies', [0.1, 0.2, 0.3])  # List of frequencies for Gabor filter
        gabor_features = [gabor(gray, frequency=freq)[0] for freq in frequencies]
        gabor_mean = np.mean(np.array(gabor_features), axis=0)
        gabor_mean = np.clip(gabor_mean, -1, 1)  
        _, gabor_thresh = cv2.threshold(img_as_ubyte(gabor_mean), self.params['texture_thresh'], 255, cv2.THRESH_BINARY)

        # Morphological Refinements
        kernel_size = self.params.get('morph_kernel_size', 5)  # Size of the morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closing = cv2.morphologyEx(gabor_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return [True, closing]

    # Displaying detected smoke regions for random images
    def procIMG(self):

        img = cv2.imread(self.img_path)
        img = self.apply_clahe(img)
        img = self.anisotropic_diffusion(img)
        lb_path = f"{self.params['trainedPath']}{self.fname}"
        if not os.path.isfile(lb_path):
            lb_path = f"{self.params['trainedPath']}test.png"
        top_hat, bottom_hat = self.apply_tophat_bothat_transforms(img)
        edges = self.multi_scale_edge_detection(img)
        lbp_features = self.extract_lbp_features(img)

        # Processing smoke detection
        detected_smoke_mask = self.scanIMG(lb_path)
        if detected_smoke_mask[0]:
            img = cv2.imread(self.img_path)

            # Creating a red overlay for detected smoke regions
            red_overlay = cv2.merge([np.zeros_like(detected_smoke_mask[1]), np.zeros_like(detected_smoke_mask[1]), detected_smoke_mask[1]])
            detected_image = cv2.addWeighted(img, 0.7, red_overlay, 0.3, 0)
            
            _, im_arr = cv2.imencode('.jpg', detected_image)  # im_arr: image in Numpy one-dim array format.
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)

            return [True, im_b64]

        else:
            return [False, None]

if __name__=="__main__":
    sample = detectIMG("./test.png")
    result = sample.procIMG()
    if result[0]:
        cv2.imshow("Detected Smoke", result[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No smoke detected in the image.")
