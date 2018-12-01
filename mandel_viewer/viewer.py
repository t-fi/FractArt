import sys
import sdl2
import sdl2.ext
from mandel_viewer.renderer import mandel, mandel_distance

sdl2.ext.init()
window = sdl2.ext.Window("Huiii", size=(1024, 1024))
window.show()

windowSurf = sdl2.SDL_GetWindowSurface(window.window)
windowArray = sdl2.ext.pixels3d(windowSurf.contents)

windowArray[:, :, 0].fill(000)  # B
windowArray[:, :, 1].fill(000)  # G
windowArray[:, :, 2].fill(255)  # R
windowArray[:, :, 3].fill(128)  # A

center = [0., 0.]
zoom = 2.

while True:
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_QUIT:
            quit()
        if event.type == sdl2.SDL_KEYDOWN:
            if event.key.keysym.sym == sdl2.SDLK_UP:
                center[1] -= zoom / 10
            elif event.key.keysym.sym == sdl2.SDLK_DOWN:
                center[1] += zoom / 10
            elif event.key.keysym.sym == sdl2.SDLK_LEFT:
                center[0] -= zoom / 10
            elif event.key.keysym.sym == sdl2.SDLK_RIGHT:
                center[0] += zoom / 10
            if event.key.keysym.sym == sdl2.SDLK_PLUS:
                zoom /= 1.1
            if event.key.keysym.sym == sdl2.SDLK_MINUS:
                zoom *= 1.1
    windowArray[:, :, :3] = mandel(center, zoom)
    # windowArray[:, :, :3] = mandel_distance(center, zoom)
    window.refresh()
