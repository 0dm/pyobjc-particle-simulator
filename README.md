# particle-simulator 

500k-particle gravity sim rendered to a raw pixel buffer in a native macOS window. PyObjC, NumPy, and Core Graphics.

Default colour map is a Lorenz attractor heatmap. Supply `--image` to use your own bitmap instead.




https://github.com/user-attachments/assets/ad011e93-0971-49d5-b658-7b3ec2c5c6d7


## Run

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Optional flags:

```
python main.py --gpu            # Metal acceleration via MLX (pip install mlx)
python main.py --image pic.png  # custom bitmap colours (pip install Pillow)
```

## Keys

| Key | Action |
|-----|--------|
| Space | Explosion — blast particles outward |
| R | Reset simulation |
| G | Toggle gravity / Lorenz flow |
| Q / Cmd+Q | Quit |
