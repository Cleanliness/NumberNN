import pyglet
from pyglet import shapes
from pyglet.window import mouse


class Painter:

    def __init__(self):
        self.window = pyglet.window.Window()
        self.grid = []

        self.batch = pyglet.graphics.Batch()
        self.pix_len = 12

        self.clear()

        # setting up gui of painter
        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, button, modifiers):
            if (button == mouse.LEFT) and (x in range(0, 28*self.pix_len)) and (y in range(0, 28*self.pix_len)):

                # TODO make the brush bigger (circular)
                self.grid[x // self.pix_len][y // self.pix_len].color = (0, 0, 0)

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.C:
                self.clear()

        pyglet.app.run()

    def clear(self):
        self.grid = []

        # setting up rectangles and grid lines
        for x in range(0, 28):
            self.grid.append([])

            for y in range(0, 28):
                rec = shapes.Rectangle(x * self.pix_len, y * self.pix_len, self.pix_len, self.pix_len, (255, 255, 255),
                                       batch=self.batch)
                self.grid[x].append(rec)

    def convert_to_list(self):
        brightness = []
        for column in self.grid:
            for rect in column:
                brightness.append(-1 * rect.color[0] + 255)

    def export_grid(self):
        pass

p = Painter()

