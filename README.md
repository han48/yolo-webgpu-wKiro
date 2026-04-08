# YOLO Image Detection

Ứng dụng web tĩnh nhận diện đối tượng trực tiếp trên trình duyệt, sử dụng model YOLOv12 tùy chỉnh thông qua [ONNX Runtime Web](https://onnxruntime.ai/). Không cần backend — toàn bộ inference chạy 100% phía client.

## Tính năng

- Hỗ trợ nhiều model — chọn model từ dropdown, tải tự động từ `models/registry.json`
- Chọn ảnh từ máy tính (PNG, JPG, WEBP) hoặc dùng webcam real-time
- Nhận diện đối tượng với bounding box màu riêng cho từng class
- Bảng thống kê số lượng và confidence trung bình theo class
- Kính lúp (magnifier) phóng to vùng ảnh theo con trỏ chuột (×1–×5, kích thước 100–300px)
- Hiển thị thời gian inference (ms) và FPS khi dùng webcam
- Capture kết quả nhận diện vào clipboard (PNG) bằng một click
- Responsive, hoạt động trên Chrome, Firefox, Edge

## Cấu trúc dự án

```
.
├── index.html              # Toàn bộ app (HTML + CSS + JS)
├── download_model.py       # Script download + convert model YOLO → ONNX
├── requirements.txt        # Python dependencies
├── models/
│   ├── registry.json       # Danh sách model
│   └── candy/
│       ├── model.onnx
│       └── classes.txt
├── src/                    # Module tách ra để test
│   ├── validation.js
│   ├── preprocessor.js
│   ├── detector.js
│   ├── nms.js
│   ├── renderer.js
│   └── ui.js
├── tests/
│   └── property.test.js    # Property-based tests (fast-check)
└── package.json
```

## Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Model input | 640×640 (letterbox padding) |
| Output tensor | `[1, 4+N, 8400]` (N = số class) |
| Confidence threshold | 0.25 |
| IoU threshold (NMS) | 0.45 |
| Zoom kính lúp | ×1 – ×5 (bước 0.5) |
| Kích thước kính lúp | 100–300px (mặc định 180px) |
| FPS cập nhật | mỗi 500ms |
| Danh sách class | Xem `models/<model-id>/classes.txt` |

Để thêm model mới, dùng script `download_model.py` (khuyến nghị) hoặc tạo thủ công:

## Thêm model mới

### Cách 1: Dùng script (khuyến nghị)

```bash
pip install -r requirements.txt
python download_model.py <url> "<model_name>"

# Ví dụ
python download_model.py https://example.com/best.pt "My Custom Model"
```

Script sẽ tự động:
- Download file `.pt` từ URL
- Convert sang ONNX (imgsz=640)
- Tạo `classes.txt` từ metadata model
- Cập nhật `models/registry.json`

### Cách 2: Thủ công

Tạo subfolder trong `models/` và đăng ký vào `models/registry.json`:

```json
{
  "models": [
    {
      "id": "candy",
      "name": "Candy Detection",
      "modelPath": "models/candy/model.onnx",
      "classesPath": "models/candy/classes.txt"
    },
    {
      "id": "my-model",
      "name": "My Custom Model",
      "modelPath": "models/my-model/model.onnx",
      "classesPath": "models/my-model/classes.txt"
    }
  ]
}
```

---

## Chạy ở localhost

Vì app load file `model.onnx` và `classes.txt` qua `fetch`, bạn cần một HTTP server — **không thể mở `index.html` trực tiếp bằng `file://`**.

### Cách 1: Python (không cần cài thêm)

```bash
# Python 3
python3 -m http.server 8080
```

Truy cập: [http://localhost:8080](http://localhost:8080)

### Cách 2: Node.js với `serve`

```bash
npx serve .
```

### Cách 3: VS Code Live Server

Cài extension [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer), click **Go Live** ở thanh trạng thái.

### Chạy tests

```bash
npm install
npm test
```

---

## Webcam

Chuyển sang tab **Webcam**, nhấn **Bật Webcam** để yêu cầu quyền camera. Sau đó nhấn **Bắt đầu nhận diện** để chạy inference liên tục theo từng frame. Nhấn **Capture → Clipboard** để copy ảnh kết quả hiện tại vào clipboard dưới dạng PNG.

> Webcam yêu cầu HTTPS hoặc `localhost`. Nếu quyền camera bị từ chối, trình duyệt sẽ hiển thị thông báo lỗi.

---

## Deploy lên AWS (S3 + CloudFront)

### Yêu cầu

- AWS CLI đã cấu hình (`aws configure`)
- Đã tạo S3 bucket và CloudFront distribution (xem bên dưới)

### Bước 1: Tạo S3 Bucket

```bash
aws s3api create-bucket \
  --bucket yolo-app-static \
  --region ap-northeast-1 \
  --create-bucket-configuration LocationConstraint=ap-northeast-1

# Tắt public access (dùng CloudFront OAC thay thế)
aws s3api put-public-access-block \
  --bucket yolo-app-static \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

### Bước 2: Tạo CloudFront Distribution

Tạo distribution với origin là S3 bucket, bật **Origin Access Control (OAC)**. Sau đó cập nhật bucket policy để cho phép CloudFront đọc:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "cloudfront.amazonaws.com"
    },
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::yolo-app-static/*",
    "Condition": {
      "StringEquals": {
        "AWS:SourceArn": "arn:aws:cloudfront::ACCOUNT_ID:distribution/DISTRIBUTION_ID"
      }
    }
  }]
}
```

### Bước 3: Upload files

```bash
# Upload index.html và classes.txt (cache ngắn)
aws s3 sync . s3://yolo-app-static/ \
  --exclude ".git/*" \
  --exclude ".kiro/*" \
  --exclude "node_modules/*" \
  --exclude "src/*" \
  --exclude "tests/*" \
  --exclude "models/*.onnx" \
  --cache-control "max-age=3600"

# Upload model.onnx riêng với cache dài (ít thay đổi)
aws s3 cp models/model.onnx s3://yolo-app-static/models/model.onnx \
  --cache-control "max-age=604800"
```

### Bước 4: Invalidate cache sau khi cập nhật

```bash
aws cloudfront create-invalidation \
  --distribution-id EXXXXXXXXX \
  --paths "/index.html" "/models/classes.txt"
```

> Không cần invalidate `model.onnx` trừ khi model thay đổi. Nếu deploy model mới, đổi tên file (vd: `model-v2.onnx`) và cập nhật reference trong `index.html`.

### Deploy tự động với GitHub Actions

Tạo file `.github/workflows/deploy.yml`:

```yaml
name: Deploy to S3 + CloudFront
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Upload to S3
        run: |
          aws s3 sync . s3://${{ secrets.S3_BUCKET }}/ \
            --exclude ".git/*" --exclude ".github/*" --exclude ".kiro/*" \
            --exclude "node_modules/*" --exclude "src/*" --exclude "tests/*" \
            --exclude "models/*.onnx" \
            --cache-control "max-age=3600"

          aws s3 cp models/model.onnx s3://${{ secrets.S3_BUCKET }}/models/model.onnx \
            --cache-control "max-age=604800"
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: ap-northeast-1

      - name: Invalidate CloudFront cache
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ secrets.CF_DISTRIBUTION_ID }} \
            --paths "/index.html" "/models/classes.txt"
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

Thêm các secrets vào GitHub repository: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `CF_DISTRIBUTION_ID`.

### Chi phí ước tính

| Dịch vụ | Chi phí |
|---|---|
| S3 storage (~50MB model) | ~$0.002/tháng |
| CloudFront (1TB đầu miễn phí) | $0 với traffic nhỏ |
| CloudFront HTTPS requests | ~$0.01/10,000 requests |

**Tổng: < $1/tháng** cho app nội bộ hoặc demo.
