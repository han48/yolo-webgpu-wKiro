/**
 * File format validation logic extracted from index.html for testability.
 * Feature: yolo-image-detection
 */

export const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];

/**
 * Validate whether a file's MIME type is accepted by the app.
 * @param {string} mimeType
 * @returns {boolean} true if accepted, false otherwise
 */
export function isAcceptedFileType(mimeType) {
  return ACCEPTED_TYPES.includes(mimeType);
}
