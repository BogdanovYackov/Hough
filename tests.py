import numpy as np
import cv2
import pytest
import math
import random
from hough import find_circles
import multiprocessing
from PIL import Image
import time


def check_success_execution(*args, **kwargs):
    circles = find_circles(*args, **kwargs)
    assert len(circles) > 0, "No circles found in the obvious case"
    

def check_if_circle_found(real_center, found_centers, shape, correct_window=0.02):
    window_in_px = round(max(shape) * correct_window) + 2
    for center in found_centers:
        if (
            abs(real_center[0] - center[0]) <= window_in_px and 
            abs(real_center[1] - center[1]) <= window_in_px
        ):
            return True
    return False


def check_center_circle(height, width, time_limit=20):
    radius = min(height // 2, width // 2)
    image = get_drawn_circles(
        shape=(height, width), 
        centers=[(height // 2, width // 2)], 
        radius=radius, 
        thickness=2
    )
    def worker():
        check_success_execution(image, radius)
    p = multiprocessing.Process(target=worker)

    # Run testing
    p.start()
    p.join(time_limit)
    if p.is_alive():
        p.terminate()
        raise TimeoutError(f"Terminated. Time limit exceeded (>={time_limit}s). Shape {(height, width)}")


def get_drawn_circles(shape, centers, radius, thickness):
    image = np.zeros(shape, dtype=np.uint8)
    for center in centers:
        center = (center[1], center[0])
        cv2.circle(image, center, radius, 255, thickness)
    return image


class TestFormat:
#### JPEG format ####
    @staticmethod
    def test_jpeg_grayscale():
        check_success_execution('test_images/circle_template_41.jpg', 41)
        check_success_execution('test_images/circle_inversed_template_41.jpg', 41)

    @staticmethod
    def test_jpeg_color():
        check_success_execution('test_images/colored_circle_41.jpg', 41)

#### PNG format ####
    @staticmethod
    def test_png_grayscale():
        check_success_execution('test_images/circle_template_41.png', 41)
        check_success_execution('test_images/circle_inversed_template_41.png', 41)

    @staticmethod
    def test_png_color():
        check_success_execution('test_images/colored_circle_41.png', 41)

    @staticmethod
    def test_png_transparent():
        check_success_execution('test_images/transparent_circle_41.png', 41)

#### GIF format ####
    @staticmethod
    def test_gif_grayscale():
        check_success_execution('test_images/circle_template_41.gif', 41)
        check_success_execution('test_images/circle_inversed_template_41.gif', 41)

    @staticmethod
    def test_gif_color():
        check_success_execution('test_images/colored_circle_41.gif', 41)


class TestShape:
    @staticmethod
    def test_tiny():
        check_success_execution('test_images/circle_template_5.png', 5, quantile=0.9)
        check_success_execution('test_images/transparent_circle_3x3.png', 1, quantile=0.5)
        check_success_execution('test_images/one_pixel.png', 1, quantile=0)

    @staticmethod
    def test_elongated():
        check_success_execution('test_images/circle_template_5_shape_10x100.png', 5)
        check_success_execution('test_images/circle_template_5_shape_100x10.png', 5)
        check_success_execution('test_images/colored_circles_50_shape_100x1000.png', 50)
        check_success_execution('test_images/circle_template_1_shape_1x100.png', 1, quantile=0.75)


    @staticmethod
    @pytest.mark.parametrize(
            "height,width", 
            [(64, 64), (128, 128), (256, 256), (512, 512),
            (64, 128), (256, 128), (512, 1024), (2048, 64), (1024, 128)])
    def test_2k_size(height, width):
        check_center_circle(height, width)


    @staticmethod
    @pytest.mark.parametrize(
            "height,width", 
            [(64, 64), (128, 128), (256, 256), (512, 512),
            (64, 128), (256, 128), (512, 1024), (2048, 64), (1024, 128)])
    def test_2k_plus_minus_one(height, width):
        for dx in range(-1, 2):
           for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                check_center_circle(height + dx, width + dy)
 

class TestSingleCircle:
    circles = [
        (100, 200, (50, 100), 50),
        (200, 100, (50, 100), 50),
        (100, 200, (30, 100), 20),
        (100, 200, (50, 20 ), 40),
        (200, 100, (180, 50), 10),
        (200, 200, (130, 150), 15),
        (300, 300, (150, 140), 130),
        (400, 200, (350, 70 ), 70),
        (400, 400, (200, 200), 100)
    ]
    
    
    @staticmethod
    @pytest.mark.parametrize("height,width,center,radius", circles) 
    def test_find_circle(height, width, center, radius):
        '''
        Test on finding at least one correct circle. 
        The circle center must be near to the output for find_circles(...).
        The quantile is set to get one the most obvious circle.
        '''
        image = get_drawn_circles((height, width), [center], radius, 2)
        quantile = 1 - 1.1 / (height * width)
        centers = find_circles(image, radius, quantile)
        assert len(centers) > 0, "No circles found in the obvious case"
        assert check_if_circle_found(center, centers, (height, width)), \
            f"Circle {center} not found in the obvious case"
        
        
    @staticmethod
    @pytest.mark.parametrize("height,width,center,radius", circles) 
    def test_find_only_one_circle(height, width, center, radius):
        '''
        Test on finding just a one circle. 
        The circle center must be the only output for find_circles(...).
        The quantile is set to get one the most obvious circle.
        '''
        image = get_drawn_circles((height, width), [center], radius, 2)
        quantile = 1 - 1.1 / (height * width)
        centers = find_circles(image, radius, quantile)
        assert len(centers) > 0, "No circles found in the obvious case"
        assert len(centers) == 1, "Found more than one circle: " + str(centers)


    @staticmethod
    @pytest.mark.parametrize("noise_lvl", [10, 100, 255])
    @pytest.mark.parametrize("height,width,center,radius", circles)
    def test_find_noised_circle(noise_lvl, height, width, center, radius):
        image = get_drawn_circles((height, width), [center], radius, 2)
        #Noising
        image = image.astype(int)
        image += np.random.randint(-noise_lvl, noise_lvl+1, size=(height, width))
        image = np.maximum(image, 0)
        image = np.minimum(image, 255)
        image = image.astype(np.uint8)

        quantile = 1 - 1.1 / (height * width)
        centers = find_circles(image, radius, quantile)
        assert len(centers) > 0, "No circles found in the noised case"
        assert check_if_circle_found(center, centers, (height, width)), \
            f"Circle {center} not found in the noised case"


    @staticmethod
    @pytest.mark.parametrize("height,width,center,radius", circles) 
    def test_find_1px_circle(height, width, center, radius):
        image = get_drawn_circles((height, width), [center], radius, 1)
        quantile = 1 - 1.1 / (height * width)
        centers = find_circles(image, radius, quantile)
        assert len(centers) > 0, "No circles found in the obvious case"
        assert check_if_circle_found(center, centers, (height, width)), \
            f"Circle {center} not found in the obvious case"
        
    
    @staticmethod
    @pytest.mark.parametrize("height,width,center,radius", circles)
    def test_find_bold_circle(height, width, center, radius):
        image = get_drawn_circles((height, width), [center], radius, 6)
        quantile = 1 - 1.1 / (height * width)
        centers = find_circles(image, radius, quantile)
        assert len(centers) > 0, "No circles found in the obvious case"
        assert check_if_circle_found(center, centers, (height, width)), \
            f"Circle {center} not found in the obvious case"

    
    cut_circles = [
        (100, 200, (40, 100), 50),
        (200, 100, (40, 40 ), 50),
        (100, 200, (30, 100), 35),
        (100, 200, (50, 20 ), 40),
        (200, 100, (195, 95), 7),
        (200, 200, (199, 199), 40),
        (300, 300, (150, 140), 170),
        (400, 200, (350, 170), 170),
        (400, 400, (200, 200), 300)
    ]
    
    @staticmethod
    @pytest.mark.parametrize("height,width,center,radius", cut_circles)
    def test_find_cut_circle(height, width, center, radius):
        image = get_drawn_circles((height, width), [center], radius, 2)
        full_circle_pixels = 4 * math.pi * radius
        got_circle_pixels = image.sum() // 255
        circle_coverage = got_circle_pixels / full_circle_pixels * 100
        
        quantile = 1 - 1.1 / (height * width)
        centers = find_circles(image, radius, quantile)
        assert len(centers) > 0, f"No circles found, case {circle_coverage}% coverage."
        assert check_if_circle_found(center, centers, (height, width)), \
            f"Circle {center} not found, case {circle_coverage}% coverage."

##################################
#### Randomly generated tests ####
##################################

class TestRandom:
    configurations = [
        (100, 200, 30, 5, 10),
        (200, 200, 40, 2, 10),
        (20, 200, 10, 4, 5),
        (200, 500, 30, 10, 5),
        (300, 500, 50, 10, 5),
        (300, 500, 30, 20, 5)
    ]
    
    @staticmethod
    @pytest.mark.parametrize("height,width,radius,n_circles,n_runs", configurations)
    def test_find_all_circles(height, width, radius, n_circles, n_runs):
        '''
        Test on finding randomly drawn circles. 
        Each circle center must be in the output of find_circles(...)
        But there is no restriction on the number of points in the output
        (See not_many_found_test).
        '''
        for run_id in range(n_runs):
            centers = [
                (random.randint(0, height), random.randint(0, width)) # (x, y)
                for _ in range(n_circles)
            ]
            image = get_drawn_circles((height, width), centers, radius, 2)
            found_centers = find_circles(image, radius, 0.97)

            for center in centers:
                assert check_if_circle_found(center, found_centers, (height, width)), \
                f"Circle {center} not found. Shape: {(height, width)}, Radius: {radius}. Run: {run_id}."


    @staticmethod
    @pytest.mark.parametrize("height,width,radius,n_circles,n_runs", configurations)
    def test_not_many_found(height, width, radius, n_circles, n_runs):
        '''
        Test on finding randomly drawn circles without large amount of false circles. 
        There is a restriction on a maximum number of found circles.
        '''
        max_number = 2 * n_circles
        for run_id in range(n_runs):
            centers = [
                (random.randint(0, height), random.randint(0, width)) # (x, y)
                for _ in range(n_circles)
            ]

            image = get_drawn_circles((height, width), centers, radius, 2)
            found_centers = find_circles(image, radius, 0.97)
            
            assert len(found_centers) <= max_number, \
            f"Found {len(found_centers)} circles when there were only {n_circles}. Run: {run_id}."