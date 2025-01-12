# Sudoku Solver

This project is a computer vision-based Sudoku Solver that detects a Sudoku grid from an input image, extracts the digits, solves the puzzle, and displays the solution. It utilizes OpenCV for image processing and PyTesseract for digit recognition.

## Features

- Preprocesses the input image to enhance the Sudoku grid.
- Detects and extracts the Sudoku grid using contour approximation.
- Recognizes digits using Tesseract OCR.
- Solves the Sudoku puzzle using a backtracking algorithm.
- Displays the original and solved Sudoku grids side-by-side.

## Prerequisites

Before running the project, ensure the following are installed:

- Python 3.7 or higher
- OpenCV
- NumPy
- PyTesseract

### Installing Dependencies

Use the following command to install the required Python libraries:

```bash
pip install opencv-python numpy pytesseract
```

### Installing Tesseract OCR

Tesseract OCR must be installed on your system. Follow the instructions for your operating system:

- **Windows**: Download and install Tesseract from [Tesseract for Windows](https://github.com/tesseract-ocr/tesseract).
- **Linux**: Install using your package manager, e.g., `sudo apt install tesseract-ocr`.
- **MacOS**: Use Homebrew, e.g., `brew install tesseract`.

Ensure the Tesseract executable path is set in your system’s environment variables or configure it in your Python script if necessary.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sudoku-solver.git
cd sudoku-solver
```

2. Place your Sudoku image (e.g., `sudoku.png`) in the project directory.

3. Run the script:

```bash
python sudoku_solver.py
```

4. The program will process the input image, solve the Sudoku puzzle, and display the original and solved grids side-by-side.

## File Structure

- **sudoku_solver.py**: Main Python script for the Sudoku Solver.
- **sudoku.png**: Example Sudoku image (replace with your image).
- **README.md**: Documentation for the project.

## Functions

### 1. `preprocess_image(image_path)`
- Prepares the image by converting it to grayscale, applying Gaussian blur, and enhancing contrast.
- Returns a preprocessed binary image.

### 2. `find_sudoku_contour(image)`
- Detects the largest contour approximating a Sudoku grid.
- Returns the contour points of the grid.

### 3. `reorder_points(points)`
- Reorders the points of the contour for perspective transformation.

### 4. `warp_sudoku(image, contour)`
- Applies a perspective warp to isolate the Sudoku grid.

### 5. `extract_digits(warped)`
- Splits the warped grid into cells, extracts digits using PyTesseract, and returns a 9x9 Sudoku board.

### 6. `can_place(arr, num, row, col)`
- Checks if a number can be placed in a specific cell according to Sudoku rules.

### 7. `sudoku(arr, n, row, col)`
- Solves the Sudoku puzzle using a backtracking algorithm.

### 8. `draw_solution(warped, solution, board)`
- Overlays the solved numbers onto the Sudoku grid.

### 9. `main(image_path)`
- Main function that integrates all steps to preprocess the image, solve the puzzle, and display the result.

## Example Output

The program outputs a side-by-side view of the original Sudoku grid and the solved grid. Ensure the input image has clear and legible digits for accurate recognition.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


Enjoy solving Sudoku puzzles effortlessly with this project!

