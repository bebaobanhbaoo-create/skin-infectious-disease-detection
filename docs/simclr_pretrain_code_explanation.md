# Giải thích code tiền huấn luyện SimCLR

## 1. Mục tiêu

Phần tiền huấn luyện SimCLR dùng để học biểu diễn ảnh da liễu từ dữ liệu không nhãn. Mỗi ảnh gốc được tạo thành hai phiên bản augmentation khác nhau. Encoder học cách đưa hai phiên bản của cùng một ảnh lại gần nhau trong không gian embedding, đồng thời đẩy các ảnh khác nhau ra xa hơn.

Sau khi train xong, projection head không được dùng cho các bước sau. File quan trọng nhất là encoder đã học, vì encoder này có thể dùng tiếp cho k-NN evaluation, Linear Probe và Prototypical Network.

## 2. Cấu trúc thư mục

- `data/simclr_pretrain/all/`: chứa ảnh da liễu không nhãn dùng cho SimCLR.
- `configs/`: chứa file cấu hình `simclr_config.yaml`.
- `src/datasets/`: chứa dataset đọc ảnh và tạo hai view.
- `src/transforms/`: chứa augmentation SimCLR.
- `src/models/`: chứa encoder, projection head và model SimCLR.
- `src/losses/`: chứa NT-Xent loss.
- `src/train/`: chứa vòng lặp huấn luyện.
- `src/utils/`: chứa hàm hỗ trợ về seed, logging, checkpoint, config và visualization.
- `scripts/`: chứa script chạy trực tiếp.
- `checkpoints/`: lưu model đã train.
- `outputs/`: lưu log, loss, biểu đồ và ảnh kiểm tra augmentation.

## 3. Giải thích từng file

`src/datasets/simclr_dataset.py` đọc ảnh không nhãn từ `data/simclr_pretrain/all/`. Mỗi lần lấy một ảnh, dataset gọi transform hai lần độc lập để tạo `view1` và `view2`. Hai view này là positive pair trong SimCLR. Code không lưu sẵn các ảnh augmentation ra ổ cứng vì augmentation cần được sinh ngẫu nhiên trong quá trình train.

`src/transforms/simclr_transforms.py` định nghĩa chuỗi augmentation gồm random crop, horizontal flip, color jitter nhẹ, grayscale xác suất thấp, Gaussian blur nhẹ, `ToTensor` và normalize ImageNet. Các mức augmentation được giữ vừa phải để không làm mất màu sắc, kết cấu, vảy da, mụn nước, mụn mủ hoặc ranh giới tổn thương.

`src/models/backbone.py` tạo ResNet50 encoder. Nếu `pretrained: true`, model dùng weights ImageNet. Lớp fully connected cuối của ResNet50 được bỏ đi, nên output là feature vector 2048 chiều thay vì logit phân loại ImageNet.

`src/models/projection_head.py` tạo MLP gồm Linear, ReLU và Linear. Projection head chỉ phục vụ contrastive loss trong pretrain. Khi dùng cho các tác vụ sau, ta bỏ projection head và giữ encoder.

`src/models/simclr.py` ghép encoder và projection head thành model SimCLR. Luồng xử lý là `image -> encoder -> feature -> projection head -> projection vector`.

`src/losses/nt_xent_loss.py` cài đặt NT-Xent loss. Loss normalize projection vector, tính similarity giữa các mẫu trong batch, kéo hai view của cùng một ảnh lại gần nhau và đẩy các ảnh khác nhau ra xa nhau. `temperature` điều khiển độ sắc của phân phối similarity.

`src/train/train_simclr.py` chứa toàn bộ quá trình train: load dataset, tạo DataLoader, tạo model, optimizer, NT-Xent loss, train theo epoch, lưu checkpoint, lưu CSV loss và vẽ biểu đồ loss.

`src/utils/visualization.py` tạo một số ảnh mẫu gồm ảnh gốc và các view augmentation. Bước này giúp kiểm tra augmentation trước khi train thật, để tránh crop quá mạnh, đổi màu quá nhiều hoặc blur làm mất chi tiết.

`scripts/00_visualize_augmentation.py` chạy riêng bước trực quan hóa augmentation. Nên chạy script này trước khi train.

`scripts/01_train_simclr.py` chạy quá trình train SimCLR từ config. Script có tham số override như `--epochs`, `--batch-size`, `--num-workers`, `--max-steps`, `--device`, `--pretrained` và `--no-pretrained` để tiện kiểm thử nhanh.
Script cũng hỗ trợ `--resume` để train tiếp từ checkpoint full model và `--run-name` để lưu kết quả vào thư mục riêng, tránh ghi đè thí nghiệm cũ.

## 4. Luồng chạy

1. Chuẩn bị ảnh trong `data/simclr_pretrain/all/`.
2. Chạy script trực quan hóa augmentation.
3. Kiểm tra ảnh mẫu trong `outputs/figures/augmentation_samples/`.
4. Chạy script train SimCLR.
5. Dataset tạo `view1` và `view2` trong lúc train.
6. Encoder trích xuất feature.
7. Projection head tạo projection vector.
8. NT-Xent loss tính contrastive loss.
9. Optimizer cập nhật trọng số encoder và projection head.
10. Lưu encoder tốt nhất để dùng cho các bước sau.

## 5. Output sau khi train

- `checkpoints/simclr/simclr_encoder_best.pth`: encoder có loss tốt nhất, là file quan trọng nhất cho downstream task.
- `checkpoints/simclr/simclr_encoder_last.pth`: encoder ở epoch cuối.
- `checkpoints/simclr/simclr_full_model_last.pth`: encoder, projection head và optimizer ở epoch cuối.
- `outputs/results/simclr_loss.csv`: loss theo từng epoch.
- `outputs/figures/simclr_loss_curve.png`: biểu đồ loss.
- `outputs/figures/augmentation_samples/`: ảnh mẫu để kiểm tra augmentation.
- `outputs/logs/simclr_train.log`: log quá trình train.

## 6. Khái niệm chính

Encoder là mạng trích xuất đặc trưng ảnh. Trong pipeline này encoder là ResNet50 bỏ lớp phân loại cuối.

Projection head là MLP đặt sau encoder trong lúc pretrain. Nó giúp tối ưu contrastive loss, nhưng không phải biểu diễn chính dùng cho các bài toán sau.

Positive pair là hai view augmentation khác nhau của cùng một ảnh gốc.

Negative pair là các view đến từ ảnh gốc khác trong cùng batch.

NT-Xent loss là loss contrastive dùng trong SimCLR. Loss này tăng similarity của positive pair và giảm similarity của negative pair.

Temperature là tham số điều chỉnh độ mạnh yếu của similarity trước khi tính cross entropy.

Augmentation giúp model học đặc trưng bền vững hơn, không quá phụ thuộc vào một cách chụp hay một biến đổi nhỏ của ảnh.

Sau pretrain chỉ giữ encoder vì encoder là phần học biểu diễn ảnh; projection head chỉ là thành phần hỗ trợ loss trong giai đoạn SimCLR.

## 7. Hướng dẫn chạy

Kiểm tra augmentation trước khi train:

```bash
python scripts/00_visualize_augmentation.py --config configs/simclr_config.yaml
```

Train SimCLR:

```bash
python scripts/01_train_simclr.py --config configs/simclr_config.yaml
```

Khi không truyền `--run-name`, script tự tạo tên run theo thời gian, ví dụ `simclr_20260616_110530`. Output sẽ được lưu vào thư mục riêng theo tên này, nên lần chạy mới không ghi đè lần chạy cũ.

Nếu muốn tự đặt tên thí nghiệm:

```bash
python scripts/01_train_simclr.py --config configs/simclr_config.yaml --run-name simclr_run_01
```

Train tiếp sau khi dừng giữa chừng. Lưu ý resume chỉ tiếp tục từ epoch đã hoàn thành gần nhất:

```bash
python scripts/01_train_simclr.py --config configs/simclr_config.yaml --resume checkpoints/simclr/simclr_full_model_last.pth
```

Với các run timestamp hoặc run có tên riêng, có thể resume trực tiếp bằng checkpoint trong thư mục run. Script sẽ tự suy ra tên run từ đường dẫn checkpoint:

```bash
python scripts/01_train_simclr.py --config configs/simclr_config.yaml --resume checkpoints/simclr/simclr_20260616_110530/simclr_full_model_last.pth
```

Bạn vẫn có thể truyền rõ `--run-name` nếu muốn:

```bash
python scripts/01_train_simclr.py --config configs/simclr_config.yaml --run-name simclr_run_01 --resume checkpoints/simclr/simclr_run_01/simclr_full_model_last.pth
```

Kiểm thử nhanh có thể chạy:

```bash
python scripts/00_visualize_augmentation.py --config configs/simclr_config.yaml --num-samples 2 --num-views 2
python scripts/01_train_simclr.py --config configs/simclr_config.yaml --epochs 1 --batch-size 4 --num-workers 0 --max-steps 2
```
