# PCB Defect Detection

Image-based PCB defect detection system using OpenCV and adaptive thresholding techniques.

## Features

- Automated PCB defect detection using OTSU adaptive thresholding
- Detection of multiple defect types: missing holes, mouse bites, open circuits
- Interactive web interface powered by Gradio
- Performance metrics: F1 Score of 77.78%

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main detection system with UI:

```bash
python "Image-Based PCB Defect Detection/pcb_detector.py"
```

Compare different detection methods:

```bash
python "Image-Based PCB Defect Detection/compare_methods.py"
```

## Project Structure

```
Image-Based PCB Defect Detection/
├── pcb_detector.py       # Main detection system with Gradio UI
├── defectDetection.py    # Core detection algorithms
├── compare_methods.py    # Method comparison and evaluation
├── Annotations/          # XML annotation files
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   └── Open_circuit/
└── rotation/            # Rotated image datasets
```

## Performance

- Old Method (Fixed threshold): F1 Score 52.31%
- Current Method (OTSU Adaptive): F1 Score 77.78%
- Improvement: +25.47%

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- Gradio

## License

MIT
