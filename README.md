# affine-overlay
Overlay of two images using an affine transformation of corresponding points

This program overlays/blends two images (or their contours) with each 
other. For this process first three (affine) or more points 
(perspective) have to be selected at the two images (note: the point 
order must match). With _space_ you continue to the next view. At the 
end you see the result, if you want to redo: press _r_.

## Example
![Original 1](examples/images/original1.jpg)
![Overlay](examples/images/overlay.jpg)
![Original 2](examples/images/original2.jpg)

## Requirements
 * Python 2.7.x (with small modifications it should also work with 3.0)
 * OpenCV (tested with 2.4.11)
 * NumPy
