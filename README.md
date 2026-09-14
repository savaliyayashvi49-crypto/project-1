# 🌟 Premium Particle Reveal Animation

An interactive computer vision project built with Python and OpenCV that transforms static images into dynamic animations featuring mosaic dot grids, neon glows, flower petals, fireworks, and water reflections.

---

## ✨ Features

* **Dot Grid Reveal:** Smooth mosaic transition that gradually unveils the image block-by-block.
* **Neon Edge Detection:** Real-time edge filtering with multi-layered Gaussian blur for a vibrant neon glow effect.
* **Particle Systems:**
  * **Petals:** Swaying and floating flower petals with realistic physics.
  * **Fireworks:** Multi-stage sky-shot fireworks with glowing particle trails.
  * **Sparks & Blooms:** Ambient golden spark fields and glowing flower bloom animations.
* **Bottom Reflection:** Dynamic water reflection effect applied to the base of the canvas.
* **Seamless Crossfade:** Smooth transition from neon aesthetics to the full high-resolution image.

---

## 🛠️ Prerequisites

Ensure you have Python installed on your system along with the required dependencies:

```bash
pip install opencv-python numpy
🚀 Usage
Place your target image inside the project directory and name it krishna.png (or update the INPUT_IMAGE path in the script).
Run the script:
Press Q or ESC to exit the animation.
⚙️ Configuration
You can customize the visual parameters directly in the code:
ParameterDefault ValueDescription
CANVAS_W, CANVAS_H1920, 1080Render resolution dimensions
FPS90Target frame rate
MOSAIC_BLOCK_SIZE12Grid block size (smaller values yield higher detail)
PETAL_COUNT110Total falling petals on screen
FIREWORK_COUNT11Simultaneous firework rockets
Built With
Python 3
OpenCV (cv2)
NumPy
