from mpdaf.obj import Cube
from src.nai_analysis.plotting.plot_helpers import rescale_8bit
import numpy as np

def RGB_image(cubefile: str, threshold: float = 100):
    cube = Cube(cubefile)
    
    imb = cube.get_band_image('Johnson_B').data
    imv = cube.get_band_image('Johnson_V').data
    imr = cube.get_band_image('Cousins_R').data

    b = rescale_8bit(imb, cmax = 900, scale='sqrt')
    v = rescale_8bit(imv, cmax = 1000, scale='sqrt')
    r = rescale_8bit(imr, cmax = 750, scale='sqrt')

    ny, nx = imb.shape

    rgb = np.zeros([ny, nx, 3], dtype=np.uint8)
    rgb[:,:,0] = r
    rgb[:,:,1] = v
    rgb[:,:,2] = b

    # Case 1: Only one channel is 255, and the other two are low
    r_peak = (rgb[..., 0] == 255) & (rgb[..., 1] < threshold) & (rgb[..., 2] < threshold)
    g_peak = (rgb[..., 1] == 255) & (rgb[..., 0] < threshold) & (rgb[..., 2] < threshold)
    b_peak = (rgb[..., 2] == 255) & (rgb[..., 0] < threshold) & (rgb[..., 1] < threshold)

    # Case 2: Two channels are 255, and one is low
    rg_peak = (rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] < threshold) 
    rb_peak = (rgb[..., 0] == 255) & (rgb[..., 2] == 255) & (rgb[..., 1] < threshold) 
    gb_peak = (rgb[..., 1] == 255) & (rgb[..., 2] == 255) & (rgb[..., 0] < threshold)  

    mask = r_peak | g_peak | b_peak | rg_peak | rb_peak | gb_peak

    rgb[mask] = 0
    return rgb