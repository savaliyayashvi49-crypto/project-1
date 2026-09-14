import os
import random
import math
import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE = os.path.join(script_dir, "krishna.png")

CANVAS_W, CANVAS_H = 1920, 1080
FPS = 90  # Higher playback/update rate

WINDOW_NAME = "Premium Krishna Reveal"
FULLSCREEN = True


# Timeline
DOT_REVEAL_SECONDS = 0.55  # Fast main reveal
REFINE_SECONDS = 0.30  # Fast refinement
HOLD_SECONDS = 0.45  # Short neon hold
CROSSFADE_SECONDS = 0.25  # Quick transition
FINAL_HOLD_SECONDS = 1.0  # Final image stays visible
# Image 
MOSAIC_BLOCK_SIZE = 12  # Fewer blocks = faster reveal
DARK_PIXEL_SKIP = 18
RANDOM_SEED = 42

# Neon
EDGE_LOW = 60
EDGE_HIGH = 150
LINE_THICKNESS = 2
GLOW_STRENGTH = 1.6

# Reflection
REFLECTION_HEIGHT_RATIO = 0.28

# Screen
DISPLAY_BRIGHTNESS = 15
DISPLAY_CONTRAST = 1.05
AUTO_CONTRAST = True

# ---------------- PREMIUM PARTICLES ----------------

PETAL_COUNT = 110
FLOWER_COUNT = 34
FIREWORK_COUNT = 11
SPARK_COUNT = 160

PETAL_MIN_SIZE = 5
PETAL_MAX_SIZE = 13

# Gold / flower palette in BGR
GOLD = (40, 190, 255)
BRIGHT_GOLD = (80, 230, 255)
WHITE_GOLD = (180, 245, 255)

PINK = (180, 90, 255)
MAGENTA = (210, 60, 230)
RED_FLOWER = (80, 80, 255)
ORANGE_FLOWER = (40, 150, 255)
YELLOW_FLOWER = (40, 220, 255)

SKY_BLUE = (255, 130, 40)
VIOLET = (220, 90, 180)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def load_and_fit(path, w, h):
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    ih, iw = img.shape[:2]

    scale = min(w / iw, h / ih)

    new_w = int(iw * scale)
    new_h = int(ih * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2

    canvas[
        y_off:y_off + new_h,
        x_off:x_off + new_w
    ] = resized

    return canvas


def make_neon_edge_layer(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    edges = cv2.Canny(
        gray,
        EDGE_LOW,
        EDGE_HIGH
    )

    if LINE_THICKNESS > 0:

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=LINE_THICKNESS
        )

    mask = edges.astype(bool)

    color_layer = np.zeros_like(img)

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[..., 1] = np.clip(
        hsv[..., 1] * 1.7,
        0,
        255
    )

    hsv[..., 2] = np.clip(
        hsv[..., 2] * 1.5 + 60,
        0,
        255
    )

    boosted = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    color_layer[mask] = boosted[mask]

    inner_glow = cv2.GaussianBlur(
        color_layer,
        (0, 0),
        sigmaX=4,
        sigmaY=4
    )

    outer_glow = cv2.GaussianBlur(
        color_layer,
        (0, 0),
        sigmaX=14,
        sigmaY=14
    )

    combined = (
        color_layer.astype(np.float32) * 1.4
        + inner_glow.astype(np.float32) * 1.1
        + outer_glow.astype(np.float32) * 0.7
    ) * GLOW_STRENGTH

    combined = np.clip(
        combined,
        0,
        255
    )

    if AUTO_CONTRAST:

        lit_pixels = combined[
            combined > 5
        ]

        if lit_pixels.size > 0:

            ref = np.percentile(
                lit_pixels,
                99
            )

            if ref > 1:

                combined = np.clip(
                    combined * (255.0 / ref),
                    0,
                    255
                )

    return combined.astype(np.uint8)


def block_mosaic(img, block_size):

    if block_size <= 1:
        return img.copy()

    h, w = img.shape[:2]

    ph = (
        block_size - h % block_size
    ) % block_size

    pw = (
        block_size - w % block_size
    ) % block_size

    padded = np.pad(
        img,
        (
            (0, ph),
            (0, pw),
            (0, 0)
        ),
        mode="constant"
    )

    H, W = padded.shape[:2]

    gh = H // block_size
    gw = W // block_size

    reshaped = padded.reshape(
        gh,
        block_size,
        gw,
        block_size,
        3
    )

    reshaped = reshaped.transpose(
        0,
        2,
        1,
        3,
        4
    ).reshape(
        gh,
        gw,
        block_size * block_size,
        3
    )

    intensity = reshaped.sum(axis=-1)

    idx = np.argmax(
        intensity,
        axis=-1
    )

    block_colors = np.take_along_axis(
        reshaped,
        idx[..., None, None],
        axis=2
    ).squeeze(2)

    upsampled = np.repeat(
        np.repeat(
            block_colors,
            block_size,
            axis=0
        ),
        block_size,
        axis=1
    )

    return upsampled[
        :h,
        :w
    ].astype(np.uint8)


def get_block_grid(
    mosaic_img,
    block_size
):

    h, w = mosaic_img.shape[:2]

    blocks = []

    for by in range(
        0,
        h,
        block_size
    ):

        for bx in range(
            0,
            w,
            block_size
        ):

            color = mosaic_img[
                by,
                bx
            ]

            if int(color.max()) <= DARK_PIXEL_SKIP:
                continue

            bw = min(
                block_size,
                w - bx
            )

            bh = min(
                block_size,
                h - by
            )

            blocks.append(
                (
                    bx,
                    by,
                    bw,
                    bh,
                    tuple(
                        int(c)
                        for c in color
                    )
                )
            )

    return blocks


def draw_block(canvas, block):

    bx, by, bw, bh, color = block

    center = (
        bx + bw // 2,
        by + bh // 2
    )

    radius = max(
        1,
        min(bw, bh) // 2
    )

    cv2.circle(
        canvas,
        center,
        radius,
        color,
        -1,
        lineType=cv2.LINE_AA
    )


def add_reflection(frame, ratio):

    h, w = frame.shape[:2]

    refl_h = int(
        h * ratio
    )

    reflection = cv2.flip(
        frame[
            h - refl_h:h,
            :
        ],
        0
    )

    fade = np.linspace(
        0.30,
        0.0,
        refl_h
    ).reshape(
        refl_h,
        1,
        1
    )

    reflection = (
        reflection.astype(np.float32)
        * fade
    ).astype(np.uint8)

    out = frame.copy()

    start_y = h - refl_h

    blended = cv2.addWeighted(
        out[start_y:h],
        0.4,
        reflection,
        0.9,
        0
    )

    out[
        start_y:h
    ] = np.maximum(
        out[start_y:h],
        blended
    )

    return out


# ============================================================
# FLOWER / PETAL SYSTEM
# ============================================================

class Petal:

    def __init__(self, rng):

        self.x = rng.uniform(
            0,
            CANVAS_W
        )

        self.y = rng.uniform(
            -CANVAS_H,
            CANVAS_H
        )

        self.speed = rng.uniform(
            65,
            155
        )

        self.size = rng.uniform(
            PETAL_MIN_SIZE,
            PETAL_MAX_SIZE
        )

        self.rotation = rng.uniform(
            0,
            math.pi * 2
        )

        self.rotation_speed = rng.uniform(
            -2.5,
            2.5
        )

        self.swing = rng.uniform(
            20,
            75
        )

        self.phase = rng.uniform(
            0,
            math.pi * 2
        )

        self.color = rng.choice([
            PINK,
            MAGENTA,
            RED_FLOWER,
            ORANGE_FLOWER,
            YELLOW_FLOWER
        ])

    def update(self, dt, time):

        self.y += self.speed * dt

        self.x += math.sin(
            time * 1.7 + self.phase
        ) * self.swing * dt

        self.rotation += (
            self.rotation_speed * dt
        )

        if self.y > CANVAS_H + 30:

            self.y = -30

    def draw(self, layer):

        x = int(self.x)
        y = int(self.y)

        s = int(self.size)

        if (
            x < -30
            or x >= CANVAS_W + 30
            or y < -30
            or y >= CANVAS_H + 30
        ):
            return

        pts = []

        for i in range(6):

            a = (
                self.rotation
                + i * math.pi / 3
            )

            px = int(
                x + math.cos(a) * s
            )

            py = int(
                y + math.sin(a) * s * 0.65
            )

            pts.append(
                [px, py]
            )

        pts = np.array(
            pts,
            dtype=np.int32
        )

        cv2.fillPoly(
            layer,
            [pts],
            self.color
        )


def create_petals(rng):

    return [
        Petal(rng)
        for _ in range(PETAL_COUNT)
    ]


# ============================================================
# FLOWER BLOOM EFFECT
# ============================================================

class Bloom:

    def __init__(
        self,
        x,
        y,
        color,
        delay,
        rng
    ):

        self.x = x
        self.y = y
        self.color = color

        self.delay = delay
        self.life = rng.uniform(
            1.5,
            2.5
        )

        self.size = rng.uniform(
            10,
            22
        )

        self.phase = rng.uniform(
            0,
            6.28
        )

    def draw(
        self,
        layer,
        elapsed
    ):

        t = elapsed - self.delay

        if t <= 0 or t >= self.life:
            return

        p = t / self.life

        # Smooth bloom
        ease = 1 - (1 - p) ** 3

        radius = int(
            self.size * ease
        )

        alpha = int(
            255 * (1 - p)
        )

        if radius < 2:
            return

        overlay = np.zeros_like(layer)

        for i in range(6):

            angle = (
                i * math.pi / 3
                + self.phase
            )

            px = int(
                self.x
                + math.cos(angle)
                * radius
            )

            py = int(
                self.y
                + math.sin(angle)
                * radius
            )

            cv2.circle(
                overlay,
                (px, py),
                max(2, radius // 3),
                self.color,
                -1,
                cv2.LINE_AA
            )

        cv2.circle(
            overlay,
            (self.x, self.y),
            max(2, radius // 4),
            WHITE_GOLD,
            -1,
            cv2.LINE_AA
        )

        glow = cv2.GaussianBlur(
            overlay,
            (0, 0),
            8
        )

        layer[:] = cv2.addWeighted(
            layer,
            1.0,
            glow,
            alpha / 900.0,
            0
        )

        layer[:] = cv2.addWeighted(
            layer,
            1.0,
            overlay,
            alpha / 255.0,
            0
        )


def create_blooms(rng):

    blooms = []

    positions = [
        (280, 280),
        (420, 180),
        (620, 250),
        (1300, 230),
        (1500, 320),
        (1650, 190),
        (260, 620),
        (1650, 650),
        (430, 820),
        (1450, 820),
        (850, 170),
        (1100, 170),
        (180, 430),
        (1740, 450),
        (500, 950),
        (1400, 950),
        (750, 900),
        (1200, 900),
        (90, 170),
        (1820, 160),
        (80, 760),
        (1840, 760),
        (610, 80),
        (1310, 80),
        (980, 90),
        (980, 980),
        (330, 500),
        (1590, 500),
        (700, 520),
        (1220, 520),
        (560, 350),
        (1360, 350),
        (560, 700),
        (1360, 700),
    ]

    colors = [
        PINK,
        MAGENTA,
        RED_FLOWER,
        ORANGE_FLOWER,
        YELLOW_FLOWER
    ]

    for i, (x, y) in enumerate(
        positions[:FLOWER_COUNT]
    ):

        blooms.append(
            Bloom(
                x,
                y,
                colors[
                    i % len(colors)
                ],
                i * 0.13,
                rng
            )
        )

    return blooms


# ============================================================
# FIREWORK / SKY-SHOT CRACKERS
# ============================================================

class Firework:

    def __init__(
        self,
        x,
        y,
        rng,
        start_delay=0
    ):

        self.x = x
        self.y = y

        self.start_x = x

        self.start_y = (
            CANVAS_H + 20
        )

        self.target_y = y

        self.delay = start_delay

        self.launch_time = 0

        self.exploded = False

        self.particles = []

        self.color = rng.choice([
            GOLD,
            BRIGHT_GOLD,
            PINK,
            SKY_BLUE,
            VIOLET,
            WHITE_GOLD
        ])

        self.rng = rng

    def reset(self):

        self.exploded = False
        self.particles = []

    def explode(self):

        self.exploded = True

        count = self.rng.randint(
            85,
            125
        )

        for i in range(count):

            angle = (
                i / count
                * math.pi * 2
            )

            speed = self.rng.uniform(
                180,
                390
            )

            self.particles.append({
                "x": float(self.x),
                "y": float(self.y),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.rng.uniform(
                    0.7,
                    1.35
                ),
                "age": 0.0
            })

    def draw(
        self,
        layer,
        dt,
        elapsed
    ):

        if elapsed < self.delay:
            return

        local_time = elapsed - self.delay

        # ------------------------------------------------
        # Rocket launch
        # ------------------------------------------------

        if not self.exploded:

            duration = 0.82

            p = min(
                local_time / duration,
                1.0
            )

            ease = 1 - (
                1 - p
            ) ** 3

            y = (
                self.start_y
                + (
                    self.target_y
                    - self.start_y
                ) * ease
            )

            x = self.start_x

            # Rocket trail
            for i in range(22):

                trail_p = i / 14

                ty = y + (
                    trail_p * 105
                )

                alpha = int(
                    220
                    * (1 - trail_p)
                )

                cv2.circle(
                    layer,
                    (
                        int(x),
                        int(ty)
                    ),
                    max(
                        1,
                        max(1, int(5 * (1 - trail_p)))
                    ),
                    self.color,
                    -1,
                    cv2.LINE_AA
                )

            cv2.circle(
                layer,
                (int(x), int(y)),
                5,
                WHITE_GOLD,
                -1,
                cv2.LINE_AA
            )

            if p >= 1.0:

                self.explode()

            return

        # ------------------------------------------------
        # Explosion
        # ------------------------------------------------

        alive = False

        for particle in self.particles:

            particle["age"] += dt

            if (
                particle["age"]
                >= particle["life"]
            ):
                continue

            alive = True

            age = particle["age"]
            life = particle["life"]

            particle["x"] += (
                particle["vx"] * dt
            )

            particle["y"] += (
                particle["vy"] * dt
            )

            # Gravity
            particle["vy"] += (
                125 * dt
            )

            fade = 1 - age / life

            px = int(
                particle["x"]
            )

            py = int(
                particle["y"]
            )

            if (
                px < 0
                or px >= CANVAS_W
                or py < 0
                or py >= CANVAS_H
            ):
                continue

            # Trail
            tx = int(
                px
                - particle["vx"]
                * 0.035
            )

            ty = int(
                py
                - particle["vy"]
                * 0.035
            )

            cv2.line(
                layer,
                (tx, ty),
                (px, py),
                self.color,
                max(
                    1,
                    int(3 * fade)
                ),
                cv2.LINE_AA
            )

            cv2.circle(
                layer,
                (px, py),
                max(
                    1,
                    int(3 * fade)
                ),
                WHITE_GOLD,
                -1,
                cv2.LINE_AA
            )

        if not alive:

            self.delay += (
                self.rng.uniform(
                    2.5,
                    4.0
                )
            )

            self.reset()


def create_fireworks(rng):

    positions = [
        (180, 180),
        (420, 120),
        (700, 190),
        (960, 135),
        (1220, 190),
        (1500, 120),
        (1740, 180),
        (300, 360),
        (960, 330),
        (1620, 370),
        (960, 520)
    ]

    fireworks = []

    for i, (x, y) in enumerate(
        positions
    ):

        fireworks.append(
            Firework(
                x,
                y,
                rng,
                i * 0.24
            )
        )

    return fireworks


# ============================================================
# GOLDEN SPARKLE FIELD
# ============================================================

class Spark:

    def __init__(self, rng):

        self.x = rng.uniform(
            80,
            CANVAS_W - 80
        )

        self.y = rng.uniform(
            80,
            CANVAS_H - 80
        )

        self.phase = rng.uniform(
            0,
            math.pi * 2
        )

        self.speed = rng.uniform(
            1.5,
            4.0
        )

        self.size = rng.uniform(
            1,
            3.5
        )


def create_sparks(rng):

    return [
        Spark(rng)
        for _ in range(SPARK_COUNT)
    ]


def draw_sparks(
    layer,
    sparks,
    time,
    intensity=1.0
):

    for spark in sparks:

        pulse = (
            math.sin(
                time * spark.speed
                + spark.phase
            )
            + 1
        ) / 2

        alpha = (
            pulse
            * intensity
        )

        if alpha < 0.15:
            continue

        r = max(
            1,
            int(
                spark.size
                * alpha
            )
        )

        cv2.circle(
            layer,
            (
                int(spark.x),
                int(spark.y)
            ),
            r,
            GOLD,
            -1,
            cv2.LINE_AA
        )


# ============================================================
# CINEMATIC PARTICLE COMPOSITOR
# ===============================================
