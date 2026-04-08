/**
 * UIController utilities extracted from index.html for testability.
 * Feature: yolo-image-detection
 */

/**
 * Clear the result canvas and hide the stats table section.
 *
 * @param {HTMLCanvasElement} resultCanvas  — the canvas showing detection results
 * @param {HTMLElement} tableSection        — the container element for the stats table
 * @param {HTMLElement} tableBody           — the <tbody> element to clear rows from
 */
export function clearResults(resultCanvas, tableSection, tableBody) {
  // Clear the result canvas by resetting its pixel content
  const ctx = resultCanvas.getContext('2d');
  ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

  // Hide the stats table section
  tableSection.style.display = 'none';

  // Remove all table rows
  tableBody.innerHTML = '';
}
