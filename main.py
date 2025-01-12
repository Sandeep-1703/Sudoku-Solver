import cv2
import numpy as np
import pytesseract

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve contrast
    contrast = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def find_sudoku_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:  # We expect a quadrilateral
            return approx

    return None

def reorder_points(points):
    points = points.reshape(4, 2)
    new_points = np.zeros((4, 2), dtype='float32')

    s = points.sum(axis=1)
    new_points[0] = points[np.argmin(s)]  # Top-left
    new_points[2] = points[np.argmax(s)]  # Bottom-right
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]  # Top-right
    new_points[3] = points[np.argmax(diff)]  # Bottom-left

    return new_points

def warp_sudoku(image, contour):
    points = contour.reshape(4, 2)
    points = reorder_points(points)

    width, height = 450, 450
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(points.astype('float32'), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped

def extract_digits(warped):
    cells = np.vsplit(warped, 9)
    digits = []

    for i, cell in enumerate(cells):
        cell = np.hsplit(cell, 9)
        for j, c in enumerate(cell):
            cell_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            # Crop the cell slightly smaller
            h, w = cell_gray.shape
            cropped_cell = cell_gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

            # Apply adaptive thresholding
            cell_thresh = cv2.adaptiveThreshold(cropped_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2)

            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            cell_thresh = cv2.morphologyEx(cell_thresh, cv2.MORPH_CLOSE, kernel)

            # Tesseract with different psm modes
            digit = pytesseract.image_to_string(cell_thresh, config='--psm 6 outputbase digits')

            if len(digit.strip()) == 1 and digit.strip().isdigit():
                digits.append(int(digit.strip()))
            else:
                digits.append(0)

    return digits

def can_place(arr, num, row, col):
    # Check the column
    for i in range(9):
        if arr[i][col] == num:
            return False

    # Check the row
    for j in range(9):
        if arr[row][j] == num:
            return False

    # Check the 3x3 subgrid
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if arr[i][j] == num:
                return False

    return True

def sudoku(arr, n, row, col):
    if row == n:
        return True  # Solution found

    if col == n:
        return sudoku(arr, n, row + 1, 0)

    if arr[row][col] != 0:
        return sudoku(arr, n, row, col + 1)
    else:
        for k in range(1, 10):
            if can_place(arr, k, row, col):
                arr[row][col] = k

                # Move to the next column
                if sudoku(arr, n, row, col + 1):
                    return True  # Solution found

                # Backtrack
                arr[row][col] = 0

    return False  # No solution found

def draw_solution(warped, solution, board):
    cell_height = warped.shape[0] // 9
    cell_width = warped.shape[1] // 9

    for i in range(9):
        for j in range(9):
            if solution[i][j] == 0:  # Only write in blank cells
                # Calculate the position to place the text
                x = j * cell_width + cell_width // 4
                y = i * cell_height + (cell_height + 20) // 2  # Adjust for vertical centering
                cv2.putText(warped, str(board[i][j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (199, 0, 139), 4)

    return warped

def main(image_path):
    edges = preprocess_image(image_path)

    contour = find_sudoku_contour(edges)
    if contour is None:
        raise ValueError("Sudoku grid not found.")

    original_image = cv2.imread(image_path)
    warped = warp_sudoku(original_image, contour)

    digits = extract_digits(warped)

    board = np.array(digits).reshape(9, 9)
    cat = np.copy(board)
    print('\n', '-' * 15, "Extracted Sudoku Board", '-' * 15, '\n')
    for row in board:
        print(" ".join(str(num) if num != 0 else '.' for num in row))  # Use '.' for empty cells

    # Solve the Sudoku puzzle
    print('\n', '-' * 15, "Solved Sudoku Board", '-' * 15, '\n')
    if not sudoku(board, 9, 0, 0):
        print('-' * 15, "No solution exists", '-' * 15)
    else:
        # Draw the solution on the warped image
        solution_image = draw_solution(warped.copy(), cat, board)

        # Add padding to the solved image
        padding = 20  # Adjust padding as needed
        solved_image_resized = cv2.copyMakeBorder(solution_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Resize the original image to match the size of the solved image
        original_shape = original_image.shape[:2]
        original_resized = cv2.resize(original_image, (solved_image_resized.shape[1], solved_image_resized.shape[0]))

        # Concatenate the original and solved images
        combined_image = np.hstack((original_resized, solved_image_resized))

        # Display the combined image
        cv2.imshow("Original and Solved Sudoku", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main('sudoku.png')  # Use your image path here
