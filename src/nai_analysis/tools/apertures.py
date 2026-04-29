from typing import Optional
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse

class Aperture:
    """
    A simple class for storing apertures to isolate MAP values

        `a` -- the semi major axis
        `b` -- Optional, the semi minor axis

    if only `a` is supplied, the aperture is a Circle with radius `a`
    """
    def __init__(
            self, 
            label: str,
            x: float, 
            y: float, 
            a: float, 
            b: Optional[float] = None, 
            angle: float = 0., 
            isolate: Optional[str] = None,
            labelcolor: str = 'k'
        ):

        self._validate(x=x,y=y,a=a,b=b,angle=angle,label=label,isolate=isolate, labelcolor=labelcolor)

        isolate = 'none' if isolate is None else isolate

        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.angle = angle
        self.label = label
        self.isolate = isolate
        self.labelcolor = labelcolor

    @staticmethod
    def _validate(
            x: float, 
            y: float, 
            a: float, 
            b: Optional[float], 
            angle: float, 
            label: Optional[str],
            isolate: Optional[str],
            labelcolor: str
        ) -> None:
        for val in [x, y, a, angle]:
            if not isinstance(val, float | int):
                raise ValueError("Input coordinates must be float")
        if b is not None:
            if not isinstance(b, float | int):
                raise ValueError("Input coordinates must be float")
        if not isinstance(label, str):
            raise ValueError("label must be a string")
        if isolate is not None:
            if isolate.lower() not in ['outflow', 'inflow', 'none']:
                raise ValueError("isolate argument must be one of 'outflow' 'inflow' or 'none'")
        if not isinstance(labelcolor, str):
            raise ValueError("labelcolor must be a string")

    def plot_aper(self, ax: Axes, **kwargs) -> None:
        edgecolor = kwargs.get('edgecolor', 'k')
        facecolor = kwargs.get('facecolor', 'none')
        zorder = kwargs.get('zorder', 20)

        if self.b is None:
            patch = Circle((self.x, self.y), radius=self.a, edgecolor=edgecolor, facecolor=facecolor, zorder = zorder, **kwargs)

        else:
            patch = Ellipse((self.x, self.y), width=2 * self.a, height=2 * self.b, angle=self.angle)

        ax.add_patch(patch)

        if self.label is not None:
            ax.text(self.x, self.y, self.label)