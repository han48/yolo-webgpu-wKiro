# Requirements Document

## Introduction

Tính năng này xây dựng một static web app (chỉ dùng HTML, JavaScript, CSS và CDN) cho phép người dùng chọn model, chọn ảnh từ máy tính, chạy object detection trực tiếp trên trình duyệt bằng model YOLO tùy chỉnh thông qua ONNX Runtime Web (ort.js), và hiển thị kết quả nhận diện gồm ảnh gốc, ảnh có bounding box, và bảng thống kê các đối tượng phát hiện được.

App hỗ trợ nhiều model — mỗi model nằm trong một subfolder riêng dưới `models/` và được đăng ký trong file `models/registry.json`.

## Glossary

- **App**: Ứng dụng web tĩnh chạy hoàn toàn trên trình duyệt, không cần server backend.
- **Model**: File `model.onnx` trong một subfolder của `models/` — model YOLO đã được train tùy chỉnh.
- **Model_Registry**: File `models/registry.json` — danh sách các model có sẵn, mỗi entry gồm `id`, `name`, `modelPath`, `classesPath`.
- **Class_List**: File `classes.txt` trong subfolder của model — danh sách tên các class mà Model có thể nhận diện.
- **ONNX_Runtime**: Thư viện ort.js được tải qua CDN, dùng để chạy Model trực tiếp trên trình duyệt.
- **Detection_Result**: Tập hợp các bounding box, class label và confidence score trả về sau khi chạy inference.
- **Bounding_Box**: Hình chữ nhật xác định vị trí của một đối tượng trong ảnh, gồm tọa độ (x, y, width, height).
- **Confidence_Score**: Giá trị từ 0.0 đến 1.0 thể hiện độ tin cậy của một dự đoán.
- **NMS**: Non-Maximum Suppression — thuật toán lọc bỏ các bounding box trùng lặp.
- **Detection_Table**: Bảng hiển thị danh sách các class đã nhận diện, số lượng và confidence score trung bình.
- **Canvas**: Phần tử HTML canvas dùng để vẽ ảnh và bounding box.
- **Magnifier**: Kính lúp hình tròn di chuyển theo con trỏ chuột, hiển thị vùng ảnh được phóng to tại vị trí chuột.
- **Webcam**: Camera của thiết bị, dùng làm nguồn ảnh đầu vào thay thế cho file ảnh.
- **Real-time Detection**: Quá trình chạy inference liên tục trên từng frame webcam bằng `requestAnimationFrame`.
- **FPS**: Frames Per Second — số frame được xử lý mỗi giây trong chế độ webcam.
- **Inference_Time**: Thời gian (ms) để hoàn thành một lần tiền xử lý + inference + hậu xử lý.
- **Capture**: Ảnh chụp màn hình kết quả nhận diện tại một thời điểm, được lưu vào clipboard dưới dạng PNG.

---

## Requirements

### Requirement 1: Tải và khởi tạo Model

**User Story:** As a người dùng, I want chọn model từ danh sách và model được tải tự động, so that tôi có thể sử dụng đúng model mong muốn mà không cần thao tác phức tạp.

#### Acceptance Criteria

1. WHEN trang web được mở, THE App SHALL fetch file `models/registry.json` để lấy danh sách các model có sẵn.
2. WHEN danh sách model được tải, THE App SHALL hiển thị dropdown cho phép người dùng chọn model.
3. WHEN người dùng chọn model từ dropdown, THE App SHALL tải file `model.onnx` và `classes.txt` tương ứng bằng ONNX_Runtime.
4. WHILE Model đang được tải, THE App SHALL hiển thị trạng thái loading cho người dùng.
5. IF việc tải Model_Registry hoặc Model thất bại, THEN THE App SHALL hiển thị thông báo lỗi rõ ràng và không cho phép thực hiện inference.
6. WHEN Model được tải thành công, THE App SHALL hiển thị trạng thái sẵn sàng cho người dùng.
7. WHEN người dùng đổi sang model khác, THE App SHALL xóa kết quả nhận diện cũ và tải model mới.

---

### Requirement 2: Chọn ảnh đầu vào

**User Story:** As a người dùng, I want chọn ảnh từ máy tính của mình, so that tôi có thể thực hiện object detection trên ảnh bất kỳ.

#### Acceptance Criteria

1. THE App SHALL cung cấp một nút hoặc vùng kéo-thả để người dùng chọn file ảnh từ máy tính.
2. WHEN người dùng chọn file, THE App SHALL chỉ chấp nhận các định dạng ảnh PNG, JPG, JPEG và WEBP.
3. IF người dùng chọn file không phải định dạng ảnh hợp lệ, THEN THE App SHALL hiển thị thông báo lỗi định dạng.
4. WHEN người dùng chọn ảnh hợp lệ, THE App SHALL hiển thị ảnh gốc trên giao diện trước khi thực hiện inference.
5. WHEN người dùng chọn ảnh mới, THE App SHALL xóa kết quả nhận diện cũ trước khi hiển thị ảnh mới.

---

### Requirement 3: Thực hiện Object Detection

**User Story:** As a người dùng, I want nhấn nút để chạy nhận diện đối tượng trên ảnh đã chọn, so that tôi nhận được kết quả phát hiện đối tượng.

#### Acceptance Criteria

1. THE App SHALL cung cấp nút "Detect" để kích hoạt quá trình inference.
2. WHEN người dùng nhấn nút Detect, THE App SHALL tiền xử lý ảnh đầu vào về kích thước đầu vào của Model (640x640) bằng cách resize và padding.
3. WHEN người dùng nhấn nút Detect, THE App SHALL chạy inference bằng ONNX_Runtime với ảnh đã tiền xử lý.
4. WHEN inference hoàn tất, THE App SHALL áp dụng NMS để lọc các Bounding_Box trùng lặp với ngưỡng IoU là 0.45.
5. THE App SHALL chỉ giữ lại các Detection_Result có Confidence_Score lớn hơn hoặc bằng 0.25.
6. WHILE inference đang chạy, THE App SHALL hiển thị trạng thái đang xử lý và vô hiệu hóa nút Detect.
7. IF không có đối tượng nào được phát hiện, THEN THE App SHALL hiển thị thông báo "Không phát hiện đối tượng nào".

---

### Requirement 4: Hiển thị ảnh kết quả với Bounding Box

**User Story:** As a người dùng, I want xem ảnh đã được vẽ bounding box, so that tôi biết vị trí các đối tượng được phát hiện trong ảnh.

#### Acceptance Criteria

1. WHEN inference hoàn tất, THE App SHALL vẽ tất cả Bounding_Box lên Canvas hiển thị ảnh kết quả.
2. THE App SHALL vẽ mỗi Bounding_Box với màu sắc riêng biệt cho từng class.
3. THE App SHALL hiển thị nhãn gồm tên class và Confidence_Score (làm tròn 2 chữ số thập phân) bên trên mỗi Bounding_Box.
4. THE App SHALL hiển thị ảnh gốc và ảnh kết quả cạnh nhau trên giao diện.
5. THE App SHALL scale tọa độ Bounding_Box từ không gian ảnh 640x640 về kích thước ảnh gốc trước khi vẽ.

---

### Requirement 5: Hiển thị bảng thống kê kết quả

**User Story:** As a người dùng, I want xem bảng danh sách các đối tượng đã nhận diện, so that tôi biết có bao nhiêu đối tượng thuộc từng class và độ tin cậy của chúng.

#### Acceptance Criteria

1. WHEN inference hoàn tất, THE App SHALL hiển thị Detection_Table bên dưới ảnh kết quả.
2. THE Detection_Table SHALL chứa các cột: Tên Class, Số Lượng, và Confidence Score trung bình.
3. THE App SHALL tính Confidence_Score trung bình cho mỗi class từ tất cả các detection thuộc class đó.
4. THE App SHALL hiển thị Confidence_Score trung bình dưới dạng phần trăm (ví dụ: 87.5%).
5. THE Detection_Table SHALL sắp xếp các hàng theo số lượng giảm dần.
6. IF không có Detection_Result nào, THEN THE App SHALL ẩn Detection_Table.

---

### Requirement 6: Giao diện người dùng

**User Story:** As a người dùng, I want giao diện trực quan và dễ sử dụng, so that tôi có thể thao tác nhanh chóng mà không cần hướng dẫn.

#### Acceptance Criteria

1. THE App SHALL là một file HTML duy nhất với CSS và JavaScript được nhúng trực tiếp hoặc tham chiếu local.
2. THE App SHALL tải ONNX_Runtime thông qua CDN (không cần cài đặt).
3. THE App SHALL hoạt động hoàn toàn offline sau khi tải trang lần đầu (ngoại trừ CDN).
4. THE App SHALL hiển thị đúng trên các trình duyệt hiện đại (Chrome, Firefox, Edge).
5. WHERE người dùng sử dụng thiết bị có màn hình nhỏ, THE App SHALL điều chỉnh layout phù hợp (responsive).
6. THE App SHALL hiển thị dropdown chọn model ở vị trí nổi bật trên giao diện, trước khu vực chọn ảnh.

---

### Requirement 8: Nhận diện qua Webcam (Real-time)

**User Story:** As a người dùng, I want sử dụng webcam để nhận diện đối tượng theo thời gian thực, so that tôi không cần chụp và upload ảnh thủ công.

#### Acceptance Criteria

1. THE App SHALL cung cấp tab chuyển đổi giữa chế độ "Ảnh" và "Webcam".
2. WHEN người dùng chọn tab Webcam và nhấn bật, THE App SHALL yêu cầu quyền truy cập camera và hiển thị video stream.
3. WHEN webcam đang chạy, THE App SHALL cung cấp nút bắt đầu/tạm dừng nhận diện real-time.
4. WHILE nhận diện real-time đang chạy, THE App SHALL chạy inference liên tục trên từng frame và vẽ bounding box lên canvas kết quả.
5. THE App SHALL cung cấp nút dừng để tắt webcam và giải phóng camera.
6. IF quyền truy cập camera bị từ chối, THEN THE App SHALL hiển thị thông báo lỗi rõ ràng.

---

### Requirement 9: Hiển thị thời gian nhận diện và FPS

**User Story:** As a người dùng, I want xem thời gian xử lý và tốc độ nhận diện, so that tôi biết hiệu năng của model trên thiết bị của mình.

#### Acceptance Criteria

1. WHEN inference hoàn tất (cả image và webcam), THE App SHALL hiển thị thời gian nhận diện tính bằng milliseconds.
2. WHILE nhận diện webcam đang chạy, THE App SHALL hiển thị thêm chỉ số FPS (frames per second).
3. THE FPS SHALL được cập nhật định kỳ (tối thiểu mỗi 500ms) để phản ánh tốc độ thực tế.

---

### Requirement 10: Capture kết quả webcam vào Clipboard

**User Story:** As a người dùng, I want chụp ảnh kết quả nhận diện từ webcam và lưu vào clipboard, so that tôi có thể dán và chia sẻ nhanh chóng.

#### Acceptance Criteria

1. WHILE nhận diện webcam đang chạy, THE App SHALL hiển thị nút "Capture → Clipboard".
2. WHEN người dùng nhấn nút capture, THE App SHALL copy canvas kết quả nhận diện hiện tại vào clipboard dưới dạng PNG.
3. WHEN capture thành công, THE App SHALL hiển thị thông báo xác nhận.
4. IF clipboard API không khả dụng hoặc bị từ chối, THEN THE App SHALL hiển thị thông báo lỗi.

---

### Requirement 7: Kính lúp (Magnifier)

**User Story:** As a người dùng, I want phóng to vùng ảnh tôi đang xem, so that tôi có thể quan sát chi tiết bounding box và nhãn trên ảnh nhỏ.

#### Acceptance Criteria

1. WHEN người dùng di chuyển chuột lên canvas (ảnh gốc hoặc ảnh kết quả), THE App SHALL hiển thị một kính lúp hình tròn di chuyển theo con trỏ chuột.
2. THE Magnifier SHALL hiển thị nội dung canvas được phóng to tại vị trí con trỏ chuột.
3. WHEN người dùng di chuột ra khỏi canvas, THE App SHALL ẩn kính lúp.
4. THE App SHALL cung cấp thanh điều chỉnh mức độ phóng to từ ×1 đến ×5.
5. THE Magnifier SHALL tự động điều chỉnh vị trí để không bị tràn ra ngoài viewport.
6. THE Magnifier SHALL hoạt động trên cả canvas ảnh gốc lẫn canvas kết quả nhận diện.
7. THE App SHALL cung cấp thanh điều chỉnh kích thước kính lúp từ 100px đến 300px.
