#include <torch/torch.h>

#include "mtorch_common.h"



struct image  {
    int h;
    int w;
    int c;
    float *data;
};

float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

float get_pixel_extend(image m, int x, int y, int c)
{
    if (x < 0) x = 0;
    if (x >= m.w) x = m.w - 1;
    if (y < 0) y = 0;
    if (y >= m.h) y = m.h - 1;
    if (c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}

void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int ix = (int)floorf(x);
    int iy = (int)floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
        dy     * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
        (1 - dy) *   dx   * get_pixel_extend(im, ix + 1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix + 1, iy + 1, c);
    return val;
}


void place_image(image im, int w, int h, int dx, int dy, image canvas)
{
    int x, y, c;
    for (c = 0; c < im.c; ++c) {
        for (y = 0; y < h; ++y) {
            for (x = 0; x < w; ++x) {
                float rx = ((float)x / w) * im.w;
                float ry = ((float)y / h) * im.h;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(canvas, x + dx, y + dy, c, val);
            }
        }
    }
}


// C++ interface

void darkresize_forward(at::Tensor input, int w, int h, int dx, int dy, at::Tensor *output) {
  //CHECK_INPUT_CPU(input);
  //CHECK_INPUT_CPU(output);

  image im, canvas;
  im.w = input.size(0);
  im.h = input.size(1);
  im.c = input.size(2);
  im.data = input.data<float>();
  canvas.w = output->size(0);
  canvas.h = output->size(1);
  canvas.c = output->size(2);
  canvas.data = output->data<float>();
  place_image(im, w, h, dx, dy, canvas);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &darkresize_forward, "DarkResize forward (CPU)");
}
