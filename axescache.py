"""
The AxesCache class alters how an Axes instance is rendered. By default,
the normal render is recorded during the first draw. When
panning or zooming, this image is simply re-rendered at the correct
location and magnification. This permits smooth (albeit cropped and
fuzzy) panning and zooming, even when the full render is expensive.

Whenever the mouse button is released (ending a pan/zoom), the cache
is cleared and the full-axes is re-rendered
"""
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

class RenderCapture(object):

    def __init__(self, axes, renderer):
        im = self.extract_image(renderer)

        px, py, dx, dy = self._get_corners(axes)
        im = im[py[0] : py[-1] + 1, px[0] : px[-1] + 1, :]

        self.mesh = axes.pcolormesh(dx, dy[::-1], im[:, :, 0], cmap='gray')
        axes.collections.remove(self.mesh)

        self.im = AxesImage(axes, origin='upper', interpolation='nearest')
        self.im.set_data(im)
        self.im.set_extent((dx[0], dx[-1], dy[0], dy[-1]))
        self.axes = axes

    def draw(self, renderer, *args, **kwargs):
        if self.axes.get_xscale() == 'linear' and \
          self.axes.get_yscale() == 'linear':
            self.im.draw(renderer, *args, **kwargs)
        else:
            self.mesh.draw(renderer, *args, **kwargs)

    @staticmethod
    def _get_corners(axes):
        """
        Return the device and data coordinates
        for a box inset 5 pixelx from the edge
        of an axes instance

        Returns 4 1D arrays:
        px : Pixel X locations for each column of the box
        py : Pixel Y locations for each row of the box
        dx : Data X locations for each column of the box
        dy : Data Y locations for each row of the box
        """
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        pts = np.array([[xlim[0], ylim[0]],
                        [xlim[1], ylim[1]]])

        corners = axes.transData.transform(pts).astype(np.int)

        # move in 5 pixels, to avoid grabbing the tick marks
        corners[0] += 5
        corners[1] -= 5

        px = np.arange(*corners[:, 0])
        py = np.arange(*corners[:, 1])

        tr = axes.transData.inverted().transform
        dx = tr(np.column_stack((px, px)))[:, 0]
        dy = tr(np.column_stack((py, py)))[:, 1]
        return px, py, dx, dy

    @staticmethod
    def extract_image(renderer):
        result = np.frombuffer(renderer.buffer_rgba(),
                               dtype=np.uint8)
        result = result.reshape((int(renderer.height),
                                 int(renderer.width), 4)).copy()
        return result


class AxesCache(object):

    def __init__(self, axes):
        self.axes = axes

        self.overview_img = None
        self.axes.draw = self.draw
        self.refresh_on_mouseup()

    def draw(self, renderer, *args, **kwargs):
        if self.overview_img is None:
            Axes.draw(self.axes, renderer, *args, **kwargs)
            self.overview_img = RenderCapture(self.axes, renderer)
        else:
            self.axes.axesPatch.draw(renderer, *args, **kwargs)
            self.overview_img.draw(renderer, *args, **kwargs)
            self.axes.xaxis.draw(renderer, *args, **kwargs)
            self.axes.yaxis.draw(renderer, *args, **kwargs)
            for s in self.axes.spines.values():
                s.draw(renderer, *args, **kwargs)

    def reset(self):
        self.overview_img = None

    def refresh_on_mouseup(self):
        self.axes.figure.canvas.mpl_connect('button_release_event',
                                            lambda e: self.reset())


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num = 100000
    plt.scatter(np.random.randn(num), np.random.randn(num),
                s = np.random.randint(10, 50, num),
                c = np.random.randint(0, 255, num),
                alpha=.2, linewidths=0)
    cache = AxesCache(plt.gca())

    plt.show()
