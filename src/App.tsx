import { useCallback, useEffect, useRef, useState } from "react";
// const SERVER = "http://localhost:45624"
const SERVER = "https://crypto.desislav.dev:45123"

const IMAGE_SIZE = 512;
type DrawMode = "pen" | "eraser" | "bucket";
class Drawer {
  private ctx: CanvasRenderingContext2D;
  private isDrawing: boolean;
  private lastX: number;
  private lastY: number;
  private color: number[];

  private size: number;
  private type: DrawMode;
  constructor(ctx: CanvasRenderingContext2D) {
    this.ctx = ctx;
    this.isDrawing = false;
    this.lastX = 0;
    this.lastY = 0;
    this.color = Drawer.initialColor;
    this.size = 5;
    this.type = "pen";
  }
  static initialColor = [255, 0, 0, 255];
  static sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  static types = ["pen", "eraser", "bucket"] as DrawMode[];
  static get colors() {
    return [
      [0, 0, 255],
      [0, 125, 0],
      [255, 0, 0],
      [0, 191, 191],
      [191, 0, 191],
      [255, 255, 255],
    ]
  }
  public setColor(color: number[]) {
    this.color = color;
  }
  public setSize(size: number) {
    this.size = size;
  }
  public setType(type: DrawMode) {
    this.type = type;
  }
  public beginDrawing(x: number, y: number) {
    this.isDrawing = true;
    this.lastX = x;
    this.lastY = y;
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }
  public stopDrawing() {
    this.isDrawing = false;
  }
  public draw(x: number, y: number) {
    if(!this.isDrawing) {
      return;
    }
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(x, y);
    this.ctx.strokeStyle = `rgb(${this.color[0]}, ${this.color[1]}, ${this.color[2]})`;
    this.ctx.lineCap = "round"
    this.ctx.lineJoin = "round"
    this.ctx.lineWidth = this.size;
    this.ctx.stroke();
    this.lastX = x;
    this.lastY = y;
  }
  getPixel(imageData: ImageData, x: number, y: number) {
    const index = (x + y * imageData.width) * 4;
    return [
      imageData.data[index],
      imageData.data[index + 1],
      imageData.data[index + 2],
      imageData.data[index + 3]
    ];
  }
  setPixel(imageData: ImageData, x: number, y: number, color: number[]) {
    const index = (x + y * imageData.width) * 4;
    imageData.data[index] = color[0];
    imageData.data[index + 1] = color[1];
    imageData.data[index + 2] = color[2];
    imageData.data[index + 3] = color[3];
  }
  public bucketFill = (sx: number, sy: number) => {
    const imageData = this.ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const initialColor = this.getPixel(imageData, sx, sy);
    if(initialColor.every((c, i) => c === this.color[i])) {
      return;
    }
    const queue = [[sx, sy]];
    const visited = new Set();
    const toFill = [];
    while(queue.length > 0) {
      const [x, y] = queue.shift()!;
      toFill.push([x, y]);
      const currentColor = this.getPixel(imageData, x, y);
      if(currentColor.every((c, i) => c === initialColor[i])) {
        toFill.push([x, y]);
        [
          [x - 1, y],
          [x + 1, y],
          [x, y - 1],
          [x, y + 1]
        ].filter(([a, b]) => 0 <= a && a < IMAGE_SIZE && 0 <= b && b < IMAGE_SIZE)
          .filter(([a, b]) => !visited.has([a, b].join(",")))
          .forEach(([a, b]) => {
            queue.push([a, b]);
            visited.add([a, b].join(","));
          });
      }
    }
    toFill.forEach(([x, y]) => {
      this.setPixel(imageData, x, y, this.color);
    });
    this.ctx.putImageData(imageData, 0, 0);
  };
  eraser(x: number, y: number) {
    if(!this.isDrawing) {
      return;
    }
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(x, y);
    this.ctx.strokeStyle = "rgb(255, 255, 255)";
    this.ctx.lineCap = "round"
    this.ctx.lineJoin = "round"
    this.ctx.lineWidth = this.size;
    this.ctx.stroke();
    this.ctx.moveTo(x, y);
    this.lastX = x;
    this.lastY = y;

  }

  mouseDown(offsetX: number, offsetY: number) {
    if(this.type === "pen") {
      this.beginDrawing(offsetX, offsetY);
    } else if(this.type === "bucket") {
      this.bucketFill(offsetX, offsetY);
    } else {
      this.beginDrawing(offsetX, offsetY);
    }
  }
  mouseMove(offsetX: number, offsetY: number) {
    if(this.type === "pen") {
      this.draw(offsetX, offsetY);
    } else {
      this.eraser(offsetX, offsetY);
    }
  }
  mouseUp() {
    this.stopDrawing();
  }
  load(base64: string) {
    const image = new Image(IMAGE_SIZE, IMAGE_SIZE);
    image.src = base64;
    image.onload = () => {
      this.ctx.drawImage(image, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    }
  }
  clear() {
    this.ctx.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    this.ctx.fillStyle = "#fff";
    this.ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  }
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawerRef = useRef<Drawer | null>(null);
  const [real, setReal] = useState<{ src: string, label: string, predicted: string, predicted_proba: number } | undefined>(undefined)
  const [reconstructed, setReconstructed] = useState<{ src: string, predicted: string, predicted_proba: number } | undefined>(undefined)


  useEffect(() => {
    if(canvasRef.current) {
      canvasRef.current.width = IMAGE_SIZE;
      canvasRef.current.height = IMAGE_SIZE;
      const ctx = canvasRef.current.getContext("2d");
      if(ctx) {
        ctx.imageSmoothingEnabled = false
        const drawer = new Drawer(ctx);
        drawer.clear();
        const mouseDown = (e: MouseEvent) => {
          const { offsetX, offsetY } = e;
          drawer.mouseDown(offsetX, offsetY);
        }
        const mouseMove = (e: MouseEvent) => {
          const { offsetX, offsetY } = e;
          drawer.mouseMove(offsetX, offsetY);
        }
        const mouseUp = () => {
          drawer.mouseUp();
        }

        canvasRef.current.addEventListener("mousedown", mouseDown);
        canvasRef.current.addEventListener("mousemove", mouseMove);
        canvasRef.current.addEventListener("mouseup", mouseUp);
        drawerRef.current = drawer;
        return () => {
          canvasRef.current?.removeEventListener("mousedown", mouseDown);
          canvasRef.current?.removeEventListener("mousemove", mouseMove);
          canvasRef.current?.removeEventListener("mouseup", mouseUp);
        }
      }
    }
  }, []);


  const idx2cls: Record<number, string> = { 0: "HG", 1: "LG" }

  const handleGenerate = () => {
    if(drawerRef.current) {
      const canvas = canvasRef.current;
      if(canvas) {
        const data = canvas.toDataURL("image/png");
        fetch(`${SERVER}/infer`, {
          method: "POST",
          body: JSON.stringify({ img: data }),
          headers: { "content-type": "application/json" }
        }).then(r => r.json())
          .then(res => {
            setReconstructed({ src: res.img, predicted: idx2cls[res.rec_pred_label], predicted_proba: res.rec_pred_proba })
          });
      }
    }
  }
  const handleClear = () => drawerRef.current?.clear()
  const handleLoad = () => {
    fetch(`${SERVER}/example`)
      .then(r => r.json())
      .then(res => {
        setReal({ src: res.real, label: idx2cls[res.label], predicted: idx2cls[res.real_pred_label], predicted_proba: res.real_pred_proba })
        setReconstructed({ src: res.reconstructed, predicted: idx2cls[res.rec_pred_label], predicted_proba: res.rec_pred_proba })
        drawerRef.current?.load(res.segmented)
      });
  }
  return (
    <div
      style={{ display: "flex", gap: 10, flexDirection: "column", padding: 10 }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <ColorPicker
          onChange={(color) => drawerRef.current?.setColor(color)}
          colors={Drawer.colors}
          initial={Drawer.initialColor}
        />
        <div style={{ display: "flex", flexDirection: "row", gap: 10 }}>
          <ModePicker
            onChange={(mode) => drawerRef.current?.setType(mode)}
            modes={Drawer.types}
          />
          <SizePicker
            onChange={(size) => drawerRef.current?.setSize(size)}
            sizes={Drawer.sizes}
          />
        </div>
      </div>
      <div style={{ display: "flex", gap: 10 }}>
        <div style={{ display: "flex", gap: 10, flexDirection: "column" }}>
          <div>Segmentation</div>
          <canvas width={IMAGE_SIZE} height={IMAGE_SIZE} style={{ width: IMAGE_SIZE, height: IMAGE_SIZE }} ref={canvasRef}></canvas>
        </div>
        <div style={{ display: "flex", gap: 10, flexDirection: "column" }}>
          <div>Reconstructed</div>
          <img width={IMAGE_SIZE} height={IMAGE_SIZE} style={{ width: IMAGE_SIZE, height: IMAGE_SIZE }} src={reconstructed?.src}></img>
          {reconstructed && <div>Predicted: {reconstructed.predicted} {(reconstructed.predicted_proba*100).toFixed(1)}%</div>}
        </div>
        <div style={{ display: "flex", gap: 5, flexDirection: "column" }}>
          <div>Real</div>
          <img width={IMAGE_SIZE} height={IMAGE_SIZE} style={{ width: IMAGE_SIZE, height: IMAGE_SIZE }} src={real?.src}></img>
          {real &&
            <>
              <div>Predicted: {real.predicted} {(real.predicted_proba*100).toFixed(1)}%;{"    "}Real label: {real.label}</div>
            </>
          }
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "row", gap: 10 }}>
        <button onClick={handleClear}>Clear</button>
        <button onClick={handleLoad}>Load</button>
        <button onClick={handleGenerate}>Generate</button>
      </div>
      <Legend />
    </div>
  );
}

const SizePicker = ({ onChange, sizes }: { onChange: (size: number) => void, sizes: number[] }) => {
  const [size, setSize] = useState(1);
  useEffect(() => {
    setSize(sizes[0]);
  }, [sizes]);
  return (
    <div style={{ display: "flex", alignItems: 'center', flexDirection: 'column', gap: 10 }}>
      <div>{size}</div>
      <input type="range" min={1} max={sizes[sizes.length - 1]} value={size} onChange={(e) => {
        setSize(Number(e.target.value));
        onChange(Number(e.target.value));
      }} />

    </div>
  );
}

const ModePicker = ({
  onChange,
  modes,
}: {
  onChange: (mode: DrawMode) => void;
  modes: DrawMode[];
}) => {
  const [currentMode, setCurrentMode] = useState<string>(modes[0]);
  const handleChange = useCallback(
    (c: DrawMode) => {
      setCurrentMode(c);
      onChange(c);
    },
    [onChange]
  );
  const EmojiMap: Record<DrawMode, string> = {
    pen: "✏️",
    eraser: "🧹",
    bucket: "🪣",
  }
  return (
    <div style={{ flexDirection: "row", display: "flex", gap: 10 }}>
      {modes.map((mode) => (
        <div
          onClick={() => handleChange(mode)}
          key={mode}
          style={{
            width: 40,
            height: 40,
            backgroundColor: "#fff",
            borderWidth: 10,
            borderColor: mode === currentMode ? "#000" : "#fff",
            borderStyle: "solid",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            fontSize: 20
          }}
        >
          {EmojiMap[mode]}
        </div>
      ))}
    </div>
  );
};

const ColorPicker = ({
  onChange,
  colors,
  initial
}: {
  onChange: (color: number[]) => void;
  colors: number[][];
  initial: number[]
}) => {
  const [color, setColor] = useState(initial);
  const handleChange = useCallback(
    (c: number[]) => {
      setColor(c);
      onChange(c);
    },
    [onChange]
  );

  return (
    <div style={{ flexDirection: "row", display: "flex", gap: 10 }}>
      {colors.map((c) => (
        <div
          onClick={() => handleChange(c)}
          key={c.join(",")}
          style={{
            width: 40,
            height: 40,
            backgroundColor: `rgb(${c[0]}, ${c[1]}, ${c[2]})`,
            borderWidth: 10,
            borderColor: color.join(",") === c.join(",") ? "#fff" : `rgb(${c[0]}, ${c[1]}, ${c[2]})`,
            borderStyle: "solid",
          }}
        />
      ))}
    </div>
  );
};

const Legend = () => {
  return <div style={{ flexDirection: "column", display: "flex", gap: 5, marginTop: 20 }}>
    <strong>LEGEND</strong>
    {[
      { color: "rgb(0, 0, 255)", label: "Neoplastic cells" },
      { color: "rgb(0, 125, 0)", label: "Inflammatory" },
      { color: "rgb(255, 0, 0)", label: "Connective/Soft tissue cells" },
      { color: "rgb(0, 191, 191)", label: "Dead Cells" },
      { color: "rgb(191, 0, 191)", label: "Epithelial" },
      { color: "rgb(255, 255, 255)", label: "Don't care" },



    ].map(({ color, label }) => <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
      <div style={{ width: 20, height: 20, backgroundColor: color }} />
      <div>{label}</div>
    </div>)}

  </div>
}

export default App;
