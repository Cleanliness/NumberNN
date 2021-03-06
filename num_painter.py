import numpy as np
import pyglet
from pyglet import shapes
from pyglet.window import mouse
import CNN

class Painter:

    def __init__(self):
        self.window = pyglet.window.Window(640, 12*28)
        self.grid = []
        self.batch = pyglet.graphics.Batch()
        self.pix_len = 12
        self.neural_network = CNN.NN([1])
        self.neural_network.load()
        self.output_layer = []
        self.outline = []
        self.expected_val = None
        self.exp_ui = pyglet.text.Label(("N/A"),
                                  font_name='Times New Roman',
                                  font_size=20,
                                  x=420, y=12*14 - 50,
                                  anchor_x='center', anchor_y='center', batch=self.batch)

        self.clear()

        # setting up gui of painter
        @self.window.event
        def on_draw():

            self.window.clear()
            self.batch.draw()

        @self.window.event
        def on_mouse_press(x, y, button, modifiers):

            # classify clicked
            if (button == mouse.LEFT) and (x in range(350, 450)) and (y in range(293, 310)):

                img = self.convert_to_list()
                res = self.neural_network.classify(img)
                b = np.where(res == max(res))[0][0]
                self.expected_val = b
                print(b)

                self.clear()

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, button, modifiers):
            if (button == mouse.LEFT) and (x in range(1*self.pix_len, 27*self.pix_len)) and (y in range(1*self.pix_len, 27*self.pix_len)):

                m_pos = (x // self.pix_len, y // self.pix_len)

                self.grid[m_pos[0]][m_pos[1]].color = (0, 0, 0)
                r, l = self.grid[m_pos[0] + 1][m_pos[1]], self.grid[m_pos[0] - 1][m_pos[1]]
                t, b = self.grid[m_pos[0]][m_pos[1] + 1], self.grid[m_pos[0]][m_pos[1] - 1]

                # paintbrush logic
                if r.color == (255, 255, 255):
                    col = np.random.random_sample()*90 + 70
                    r.color = (col, col, col)

                if l.color == (255, 255, 255):
                    col = np.random.random_sample()*90 + 70
                    l.color = (col, col, col)

                if t.color == (255, 255, 255):
                    col = np.random.random_sample()*90 + 70
                    t.color = (col, col, col)

                if b.color == (255, 255, 255):
                    col = np.random.random_sample()*90 + 70
                    b.color = (col, col, col)

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.C:
                self.clear()

        pyglet.app.run()

    def clear(self):
        classify_label = pyglet.text.Label('Classify',
                                           font_name='Times New Roman',
                                           font_size=20,
                                           x=400, y=300,
                                           anchor_x='center', anchor_y='center', batch=self.batch)
        load_label = pyglet.text.Label('Load model',
                                       font_name='Times New Roman',
                                       font_size=20,
                                       x=420, y=250,
                                       anchor_x='center', anchor_y='center', batch=self.batch)

        tr_label = pyglet.text.Label('Train',
                                     font_name='Times New Roman',
                                     font_size=20,
                                     x=385, y=200,
                                     anchor_x='center', anchor_y='center', batch=self.batch)

        self.grid = []

        # setting up rectangles
        for x in range(0, 28):
            self.grid.append([])

            for y in range(0, 28):
                rec = shapes.Rectangle(x * self.pix_len, y * self.pix_len, self.pix_len, self.pix_len, (255, 255, 255),
                                       batch=self.batch)
                self.grid[x].append(rec)

        # setting up neuron display in output layer
        txt = []
        for i in range(0, len(self.neural_network.activations[-1])):
            act_col = int(self.neural_network.activations[-1][i] * 255 // 1)

            self.outline.append(shapes.Circle(x=560, y=12*28 - 20- 33*i, radius=14, color=(255, 255, 255), batch=self.batch))
            self.output_layer.append(shapes.Circle(x=560, y=12*28 -20 - 33*i, radius=13, color=(act_col, act_col, act_col), batch=self.batch))
            txt.append(pyglet.text.Label(str(i),
                                      font_name='Times New Roman',
                                      font_size=11,
                                      x=530, y=12*28 -20 - 33*i,
                                      anchor_x='center', anchor_y='center', batch=self.batch))

        rec = shapes.Rectangle(400, 12*14, 100, 300, color=(0, 0, 0), batch=self.batch)

        # setting up expected output
        self.exp_ui.delete()
        self.exp_ui = pyglet.text.Label(str(self.expected_val),
                                  font_name='Times New Roman',
                                  font_size=40,
                                  x=420, y=12*14- 50,
                                  anchor_x='center', anchor_y='center', batch=self.batch)


    def convert_to_list(self):
        brightness = []

        for column in self.grid:
            brightness.append([])
            for rect in column:
                brightness[-1].append(-1 * rect.color[0] + 255)

        brightness.reverse()
        brightness = np.transpose(np.array(brightness))

        nb = []
        for r in brightness:
            for p in r:
                nb.append(p)

        nb.reverse()

        for i in range(0, len(nb)//28):
            seg = nb[i*28: ((i+1)*28) + 1]

            segstr = ""
            for p in seg:
                if p > 0:
                    segstr = segstr + "#"
                else:
                    segstr = segstr + "-"
            print(segstr)
        return nb


p = Painter()

