import argparse, signal, objc, numpy as np
from Foundation import NSObject, NSMakeRect, NSTimer, NSRectToCGRect
from AppKit import (NSApplication, NSView, NSWindow, NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable, NSBackingStoreBuffered,
    NSGraphicsContext, NSApplicationActivationPolicyRegular, NSEvent,
    NSKeyDownMask, NSMenu, NSMenuItem)
from Quartz import (CGDataProviderCreateWithData, CGColorSpaceCreateDeviceRGB, CGImageCreate,
    CGContextDrawImage, kCGImageAlphaPremultipliedLast, kCGBitmapByteOrder32Big,
    kCGRenderingIntentDefault)

cli = argparse.ArgumentParser()
cli.add_argument("--gpu", action="store_true", help="Metal GPU compute via Apple MLX")
cli.add_argument("--image", type=str, help="bitmap image to sample particle colours from")
args, _ = cli.parse_known_args()

if args.gpu:
    import mlx.core as xp; _np = lambda a: np.array(a)
else:
    xp, _np = np, lambda a: a

W, H, N = 800, 600, 500_000
G, DAMP, SOFT, V0, R_INNER = 8000.0, 0.985, 10.0, 4.0, 120.0
EXPLOSION_FORCE = 12.0
PULL = 0.0004
LZ_SIGMA, LZ_RHO, LZ_BETA, LZ_DT, LZ_SCALE = 10.0, 28.0, 8/3, 0.003, 11.0


def _lorenz_bitmap(w, h, steps=500_000, sigma=10., rho=28., beta=8/3, dt=0.005):
    x, y, z = 0.1, 0.0, 0.0
    pts = np.empty((steps, 3))
    for i in range(steps):
        dx = sigma * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        pts[i] = (x, y, z)
    px = ((pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min()) * (w - 1)).astype(int)
    py = ((pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min()) * (h - 1)).astype(int)
    py = h - 1 - py
    heat = np.zeros((h, w), dtype=np.float32)
    np.add.at(heat, (py, px), 1)
    heat = np.log1p(heat)
    heat /= heat.max()
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    t = heat
    r = np.clip(1.5 * t, 0, 1)
    g = np.clip(1.5 * t - 0.6, 0, 1)
    b = np.clip(3.0 * t - 0.2, 0, 1) * (1.0 - t * 0.5)
    img[:, :, 0] = (r * 255).astype(np.uint8)
    img[:, :, 1] = (g * 255).astype(np.uint8)
    img[:, :, 2] = (b * 255).astype(np.uint8)
    return img


bmp = None
if args.image:
    from PIL import Image
    bmp = np.array(Image.open(args.image).convert("RGBA").resize((W, H)), dtype=np.uint8)
else:
    bmp = _lorenz_bitmap(W, H)


class Sim:
    def __init__(self):
        self.c = xp.array([W / 2.0, H / 2.0])
        self.buf = np.zeros((H, W, 4), dtype=np.uint8)
        self._d = None
        self.gravity_mode = False
        self.reset()

    def reset(self):
        self.p = xp.array(np.random.rand(N, 2) * [W, H])
        d = self.p - self.c
        t = xp.stack([-d[:, 1], d[:, 0]], axis=1)
        self.v = t / (xp.sqrt(xp.sum(t ** 2, axis=1, keepdims=True)) + .1) * V0
        self.lz = xp.array(np.random.randn(N, 3) * 0.1 + [0.1, 0.0, 25.0])

    def tick_gravity(self):
        dc = self.c - self.p
        d = xp.sqrt(xp.sum(dc ** 2, axis=1, keepdims=True))
        ds = xp.maximum(d, SOFT)
        gravity = dc / ds * (G / ds ** 2)
        pull = dc * PULL
        self.v = (self.v + gravity + pull) * DAMP
        self.p = self.p + self.v
        self._d = d

    def tick_lorenz(self):
        lx, ly, lz = self.lz[:, 0:1], self.lz[:, 1:2], self.lz[:, 2:3]
        dx = LZ_SIGMA * (ly - lx)
        dy = lx * (LZ_RHO - lz) - ly
        dz = lx * ly - LZ_BETA * lz
        self.lz = self.lz + xp.concatenate([dx, dy, dz], axis=1) * LZ_DT
        sx = self.lz[:, 0] * LZ_SCALE + W / 2.0
        sy = (24.0 - self.lz[:, 2]) * LZ_SCALE + H / 2.0
        self.p = xp.stack([sx, sy], axis=1)
        self._d = xp.sqrt(xp.sum((self.p - self.c) ** 2, axis=1, keepdims=True))

    def tick(self):
        if self.gravity_mode:
            self.tick_gravity()
        else:
            self.tick_lorenz()

    def explode(self):
        dc = self.p - self.c
        d = xp.sqrt(xp.sum(dc ** 2, axis=1, keepdims=True))
        ds = xp.maximum(d, 1.0)
        self.v = self.v + dc / ds * EXPLOSION_FORCE

    def draw(self):
        self.buf[:] = (0, 0, 0, 255)
        p = _np(self.p)
        xy = p.astype(np.int32)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
        x, y = xy[ok, 0], xy[ok, 1]
        self.buf[y, x] = bmp[y, x]


sim = Sim()

CG_FLAGS = kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big


def _flash_title(view, msg):
    win = view.window()
    if not win: return
    win.setTitle_(msg)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.5, view, b"revertTitle:", None, False)


class ParticleView(NSView):
    def acceptsFirstResponder(self):
        return True

    def keyDown_(self, event):
        ch = event.charactersIgnoringModifiers()
        if ch and ch.lower() == "r":
            sim.reset()
            _flash_title(self, "Reset")
        elif ch and ch == " ":
            sim.explode()
            _flash_title(self, "Explode")
        elif ch and ch.lower() == "g":
            sim.gravity_mode = not sim.gravity_mode
            if sim.gravity_mode:
                d = sim.p - sim.c
                t = xp.stack([-d[:, 1], d[:, 0]], axis=1)
                sim.v = t / (xp.sqrt(xp.sum(t ** 2, axis=1, keepdims=True)) + .1) * V0
                _flash_title(self, "Gravity")
            else:
                sim.lz = xp.array(np.random.randn(N, 3) * 0.1 + [0.1, 0.0, 25.0])
                _flash_title(self, "Lorenz")
        elif ch and ch.lower() == "q":
            NSApplication.sharedApplication().terminate_(None)

    def revertTitle_(self, _):
        win = self.window()
        if win: win.setTitle_("window")

    def updatePhysics_(self, _):
        sim.tick(); sim.draw(); self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        ctx = NSGraphicsContext.currentContext().CGContext()
        prov = CGDataProviderCreateWithData(None, sim.buf.data, W * H * 4, None)
        if not prov: return
        img = CGImageCreate(W, H, 8, 32, W * 4, CGColorSpaceCreateDeviceRGB(),
                            CG_FLAGS, prov, None, False, kCGRenderingIntentDefault)
        if img:
            CGContextDrawImage(ctx, NSRectToCGRect(self.bounds()), img)


class AppDelegate(NSObject):
    def applicationShouldTerminateAfterLastWindowClosed_(self, _):
        return True

    def applicationDidFinishLaunching_(self, _):
        menubar = NSMenu.alloc().init()
        appMenu = NSMenu.alloc().init()
        appMenu.addItemWithTitle_action_keyEquivalent_("Quit", "terminate:", "q")
        item = NSMenuItem.alloc().init(); item.setSubmenu_(appMenu)
        menubar.addItem_(item)
        NSApplication.sharedApplication().setMainMenu_(menubar)

        r = NSMakeRect(0, 0, W, H)
        mask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            r, mask, NSBackingStoreBuffered, False)
        self.window.setTitle_("window"); self.window.center()
        self.view = ParticleView.alloc().initWithFrame_(r)
        self.window.setContentView_(self.view)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1 / 60, self.view, objc.selector(self.view.updatePhysics_, signature=b"v@:@"),
            None, True)
        self.window.makeKeyAndOrderFront_(None)


if __name__ == "__main__":
    from PyObjCTools import AppHelper
    signal.signal(signal.SIGINT, lambda *_: NSApplication.sharedApplication().terminate_(None))
    app = NSApplication.sharedApplication()
    d = AppDelegate.alloc().init(); app.setDelegate_(d)
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    app.activateIgnoringOtherApps_(True)
    AppHelper.runEventLoop(installInterrupt=False)
