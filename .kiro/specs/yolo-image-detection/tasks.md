# Implementation Plan: YOLO Image Detection

## Overview

Xây dựng static web app (single HTML file) thực hiện object detection bằng YOLO qua ONNX Runtime Web. Hỗ trợ nhiều model thông qua `models/registry.json`. Triển khai theo thứ tự: cấu trúc HTML/CSS → load registry + dropdown model → load model → tiền xử lý ảnh → inference + NMS → render kết quả.

## Tasks

- [x] 1. Tạo file HTML với cấu trúc UI và CSS
  - Tạo file `index.html` với layout gồm: dropdown chọn model, khu vực chọn ảnh, nút Detect, hai canvas (ảnh gốc + kết quả), bảng thống kê, và element `#status`
  - Viết CSS inline: responsive layout (flexbox/grid), style cho dropdown, button, canvas, table, status message (màu đỏ cho lỗi)
  - Thêm thẻ `<script>` CDN cho ONNX Runtime Web (`ort.js`)
  - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.6_

- [x] 11. Implement Model Registry và Model Selector dropdown
  - [x] 11.1 Tạo file `models/registry.json` với cấu trúc `{ "models": [{ "id", "name", "modelPath", "classesPath" }] }` — entry đầu tiên là model `candy`
    - Di chuyển `models/model.onnx` và `models/classes.txt` vào `models/candy/`
    - _Requirements: 1.1_
  - [x] 11.2 Implement `loadRegistry(registryPath)` — fetch và parse `models/registry.json`, trả về `ModelEntry[]`
    - IF registry load thất bại, hiển thị error và disable toàn bộ UI
    - _Requirements: 1.1, 1.5_
  - [x] 11.3 Implement `populateModelDropdown(models)` — render `<select>` với các option từ registry, mỗi option hiển thị `name` và value là index
    - _Requirements: 1.2, 6.6_
  - [x] 11.4 Implement change handler cho dropdown — khi người dùng chọn model mới: gọi `clearResults()`, load model và classes tương ứng, cập nhật `session` và `classes`
    - _Requirements: 1.3, 1.4, 1.6, 1.7_

- [x] 1. Tạo file HTML với cấu trúc UI và CSS
  - Tạo file `index.html` với layout gồm: khu vực chọn ảnh, nút Detect, hai canvas (ảnh gốc + kết quả), bảng thống kê, và element `#status`
  - Viết CSS inline: responsive layout (flexbox/grid), style cho button, canvas, table, status message (màu đỏ cho lỗi)
  - Thêm thẻ `<script>` CDN cho ONNX Runtime Web (`ort.js`)
  - _Requirements: 6.1, 6.2, 6.4, 6.5_

- [x] 2. Implement ModelLoader và UIController
  - [x] 2.1 Implement `loadModel(modelPath)` và `loadClasses(classesPath)` — load `models/model.onnx` qua `ort.InferenceSession.create()` và fetch `models/classes.txt`
    - Hiển thị loading state khi đang tải, ready state khi thành công
    - Disable nút Detect nếu load thất bại, hiển thị error trong `#status`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 2.2 Implement `setStatus(state, message)` và `clearResults()` trong UIController
    - Quản lý các state: `loading`, `ready`, `processing`, `error`
    - `clearResults()` xóa canvas kết quả và ẩn bảng thống kê
    - _Requirements: 1.3, 1.5, 2.5, 3.6_

- [x] 3. Implement xử lý chọn ảnh
  - [x] 3.1 Implement file input handler — validate MIME type (`image/png`, `image/jpeg`, `image/webp`), hiển thị ảnh gốc lên canvas, gọi `clearResults()` khi chọn ảnh mới
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 3.2 Viết property test cho file format validation (Property 1)
    - **Property 1: File format validation**
    - Generate random MIME type strings, verify chỉ accept `image/png`, `image/jpeg`, `image/webp`
    - **Validates: Requirements 2.2, 2.3**

- [x] 4. Implement ImagePreprocessor
  - [x] 4.1 Implement `preprocessImage(imageElement)` — resize + letterbox pad ảnh về 640×640, trả về `Float32Array` tensor `[1, 3, 640, 640]` normalized `[0, 1]` cùng `scaleX`, `scaleY`, `padX`, `padY`
    - Dùng offscreen canvas để đọc pixel data
    - _Requirements: 3.2_
  - [x] 4.2 Viết property test cho preprocessing output shape và normalization (Property 2)
    - **Property 2: Preprocessing output shape and normalization**
    - Generate random image dimensions (1–4000px), verify tensor length = 1,228,800 và mọi giá trị trong [0, 1]
    - **Validates: Requirements 3.2**

- [x] 5. Implement NMS và confidence filtering
  - [x] 5.1 Implement `computeIoU(boxA, boxB)` và `applyNMS(detections, iouThreshold)` với IoU threshold 0.45
    - _Requirements: 3.4_
  - [x] 5.2 Viết property test cho NMS (Property 4)
    - **Property 4: NMS removes high-overlap boxes**
    - Generate random overlapping boxes, verify sau NMS không có cặp cùng class có IoU > 0.45
    - **Validates: Requirements 3.4**
  - [x] 5.3 Implement confidence threshold filter — chỉ giữ detections có `confidence >= 0.25`
    - _Requirements: 3.5_
  - [x] 5.4 Viết property test cho confidence filtering (Property 3)
    - **Property 3: Confidence threshold filtering**
    - Generate random detection lists, verify output chỉ chứa detections có confidence >= 0.25
    - **Validates: Requirements 3.5**

- [x] 6. Implement Detector — parse output tensor và coordinate scaling
  - [x] 6.1 Implement parse logic cho YOLO output tensor `[1, 14, 8400]`: extract `cx, cy, w, h`, tính `confidence = max(classScores)`, `classIndex = argmax(classScores)`
    - _Requirements: 3.3_
  - [x] 6.2 Implement coordinate scaling từ không gian 640×640 về ảnh gốc dùng `scaleX`, `scaleY`, `padX`, `padY`
    - _Requirements: 4.5_
  - [x] 6.3 Viết property test cho bounding box coordinate scaling (Property 5)
    - **Property 5: Bounding box coordinate scaling**
    - Generate random image sizes và box coordinates, verify scaled coordinates nằm trong bounds ảnh gốc và mapping chính xác
    - **Validates: Requirements 4.5**
  - [x] 6.4 Implement `runDetection(session, tensor, classes, confidenceThreshold, iouThreshold)` — wire parse → filter → NMS → scale, trả về `Detection[]`
    - _Requirements: 3.3, 3.4, 3.5_

- [x] 7. Checkpoint — Đảm bảo logic core hoạt động đúng
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Renderer — vẽ bounding box và bảng thống kê
  - [x] 8.1 Implement `drawDetections(canvas, image, detections, classColors)` — vẽ ảnh lên canvas, vẽ bounding box với màu theo class, hiển thị label `"ClassName: 0.XX"` bên trên mỗi box
    - Khởi tạo `classColors` map với 10 màu HSL cố định theo design
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 8.2 Viết property test cho detection rendering completeness (Property 6)
    - **Property 6: Detection rendering completeness**
    - Generate random detection lists, verify mỗi detection có label chứa class name và confidence rounded đúng 2 decimal places
    - **Validates: Requirements 4.1, 4.2, 4.3**
  - [x] 8.3 Implement `renderTable(detections)` — tổng hợp `ClassStats`, tính avg confidence, sort by count descending, render HTML table, hiển thị avg confidence dạng `"87.5%"`
    - Ẩn table nếu detections rỗng
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  - [x] 8.4 Viết property test cho detection table correctness (Property 8)
    - **Property 8: Detection table correctness**
    - Generate random detection lists với nhiều classes, verify rows sorted by count descending và avg confidence hiển thị đúng dạng percentage
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5**

- [x] 9. Wire toàn bộ flow vào Detect button handler
  - [x] 9.1 Implement click handler cho nút Detect: gọi `preprocessImage` → `runDetection` → `drawDetections` → `renderTable`, quản lý processing state và error handling
    - Disable nút khi đang xử lý, re-enable sau khi xong hoặc lỗi
    - Hiển thị "Không phát hiện đối tượng nào" nếu detections rỗng
    - _Requirements: 3.1, 3.6, 3.7, 4.4_
  - [x] 9.2 Viết property test cho results cleared on new image selection (Property 7)
    - **Property 7: Results cleared on new image selection**
    - Generate random sequences of image selections, verify previous results không còn trong DOM sau khi chọn ảnh mới
    - **Validates: Requirements 2.5**

- [x] 10. Final checkpoint — Kiểm tra toàn bộ app
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement Magnifier (kính lúp)
  - [x] 12.1 Thêm element `#magnifier` (div hình tròn, `position: fixed`, `pointer-events: none`) và `#magnifier-canvas` vào HTML; thêm CSS cho lens và zoom control bar
    - _Requirements: 7.1, 7.4_
  - [x] 12.2 Implement `initMagnifier()` — gắn `mouseenter`, `mouseleave`, `mousemove` lên cả hai canvas; tính đúng tỉ lệ CSS pixel vs canvas logical pixel khi sample vùng phóng to
    - Hiện lens khi `mouseenter`, ẩn khi `mouseleave`
    - Dùng `drawImage` để render vùng source vào magnifier canvas với zoom
    - _Requirements: 7.1, 7.2, 7.3, 7.6_
  - [x] 12.3 Implement logic tránh tràn viewport — nếu lens vượt cạnh phải/dưới thì đảo sang trái/trên con trỏ
    - _Requirements: 7.5_
  - [x] 12.4 Implement zoom control bar — `<input type="range">` min=1 max=5 step=0.5, cập nhật label `×N` realtime
    - _Requirements: 7.4_
  - [x] 12.5 Implement lens size control bar — `<input type="range">` min=100 max=300 step=10 (mặc định 180), cập nhật kích thước lens và magnifier canvas realtime
    - _Requirements: 7.7_

- [x] 13. Implement Webcam real-time detection
  - [x] 13.1 Thêm tab UI "Ảnh / Webcam" — chuyển đổi hiển thị image panel và webcam panel, dừng webcam khi chuyển về tab Ảnh
    - _Requirements: 8.1_
  - [x] 13.2 Implement `startWebcam()` — gọi `getUserMedia`, hiển thị video stream, enable các nút điều khiển
    - _Requirements: 8.2, 8.6_
  - [x] 13.3 Implement `toggleWebcamDetection()` và `webcamLoop()` — vòng lặp `requestAnimationFrame` chạy inference liên tục: vẽ frame lên original-canvas → preprocess → runDetection → vẽ bounding box lên result-canvas → renderTable
    - _Requirements: 8.3, 8.4_
  - [x] 13.4 Implement `stopWebcam()` — dừng stream, giải phóng camera, reset UI
    - _Requirements: 8.5_
  - [x] 13.5 Implement `preprocessFromCanvas(srcCanvas)` — tiền xử lý trực tiếp từ HTMLCanvasElement thay vì HTMLImageElement
    - _Requirements: 8.4_

- [x] 14. Implement timing display và FPS
  - [x] 14.1 Thêm `#timing-bar` vào HTML — hiển thị inference time (ms) và FPS (chỉ khi webcam)
    - _Requirements: 9.1, 9.2_
  - [x] 14.2 Implement `showTiming(ms, fps?)` — cập nhật timing bar sau mỗi lần inference; tính FPS mỗi 500ms trong webcam loop
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 15. Implement Capture → Clipboard
  - [x] 15.1 Implement `captureToClipboard()` — dùng `canvas.toBlob()` + `ClipboardItem` để copy result-canvas vào clipboard dưới dạng PNG; hiển thị thông báo thành công/lỗi
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

## Notes

- Tasks đánh dấu `*` là optional, có thể bỏ qua để ra MVP nhanh hơn
- Testing dùng [fast-check](https://github.com/dubzzz/fast-check) cho property-based tests
- Mỗi property test chạy tối thiểu 100 iterations
- Toàn bộ logic chạy trong browser, không có backend
