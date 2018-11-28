import sys
import sdl2
import sdl2.ext


sdl2.ext.init()
window = sdl2.ext.Window("The Pong Game", size=(1024, 1024))
window.show()

windowSurf = sdl2.SDL_GetWindowSurface(window.window)
windowArray = sdl2.ext.pixels3d(windowSurf.contents)

windowArray[:, :, 0].fill(000)  # B
windowArray[:, :, 1].fill(000)  # G
windowArray[:, :, 2].fill(255)  # R
windowArray[:, :, 3].fill(128)  # A

while True:
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_QUIT:
            quit()
        if event.type == sdl2.SDL_KEYDOWN:
            if event.key.keysym.sym == sdl2.SDLK_UP:
            elif event.key.keysym.sym == sdl2.SDLK_DOWN:
            elif event.key.keysym.sym == sdl2.SDLK_LEFT:
            elif event.key.keysym.sym == sdl2.SDLK_RIGHT:

            if event.key.keysym.sym == sdl2.SDLK_PLUS
    window.refresh()