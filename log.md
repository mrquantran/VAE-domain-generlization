## Phương pháp

Sử dụng mô hình **Variational Autoencoder (VAE)** cho bài toán **domain generalization** trên bộ dữ liệu đa miền như **PACS**, gồm 4 miền khác nhau (Photo, Art Painting, Cartoon, Sketch).

Cách tiếp cận gồm các bước sau:

1. **Chọn một miền cụ thể** (ví dụ: **Cartoon**), sử dụng một **encoder** để chuyển đổi hình ảnh từ miền đó vào **latent space**.
2. Tại latent space này, sử dụng **3 decoder riêng biệt**, mỗi decoder tương ứng với một trong **3 miền còn lại** (Photo, Art Painting, Sketch), để **tái tạo hình ảnh** trong các miền đó từ biểu diễn tiềm ẩn.
3. **Tại bottleneck** (giữa encoder và các decoder), tích hợp một **mô hình phân loại (classifier)** để **xác định class** của hình ảnh đầu vào.
4. **Hàm loss function** của bạn bao gồm hai thành phần chính:
    - **Loss tái tạo hình ảnh (Reconstruction Loss)**: đo lường sự khác biệt giữa hình ảnh tái tạo qua decoder và hình ảnh mong muốn trong các miền mục tiêu.
    - **Loss phân loại (Classification Loss)**: đánh giá độ chính xác của classifier trong việc dự đoán **lớp** của hình ảnh đầu vào.
5. **Lặp lại quá trình huấn luyện** này cho **từng miền** trong bộ dữ liệu. Ví dụ, khi chuyển sang miền **Sketch**, tiếp tục sử dụng encoder để mã hóa hình ảnh Sketch vào không gian tiềm ẩn, và sử dụng các decoder để tái tạo hình ảnh tương ứng trong **3 miền còn lại**.
6. **Mục tiêu chính** của là tạo ra một **latent space** mà ở đó các **đặc trưng chung giữa các miền** được biểu diễn **tương tự nhau**. Điều này dựa trên giả định rằng mặc dù các miền có thể khác nhau về phong cách hoặc hình thức biểu diễn, nhưng chúng vẫn **chia sẻ một số đặc trưng cơ bản** trong không gian tiềm ẩn.

Bằng cách này, tôi hy vọng mô hình sẽ học được các **đặc trưng chung (shared features)** giữa các miền, giúp cải thiện khả năng **tổng quát hóa miền (domain generalization)** khi áp dụng cho các dữ liệu mới hoặc miền chưa thấy.

## Giải thích code

### Step 1: Setting Up the Environment

The function `set_random_seeds()` is used to ensure reproducibility by setting seeds for various libraries like `torch`, `numpy`, and `random`. This is a good practice for consistent results.

### Step 2: Data Preparation - Dataset Class (`PACSDataset`)

- This class is designed to load images and their labels from a specified domain within the PACS dataset. It uses transformations to resize and normalize images, which is essential for feeding data into neural networks.
- The `get_dataloader()` function creates a DataLoader for a given domain, which is crucial for batching and shuffling data during training.

### Step 3: Model Architecture

- **Encoder:** The `Encoder` class uses to extract features from images. It then projects these features into a latent space using two linear layers (`fc_mu` and `fc_logvar`) to produce the mean and log variance for the latent distribution. This aligns with the VAE approach where the encoder maps inputs to a latent space.
- Decoder: The `Decoder` class reconstructs images from the latent space. It includes a domain embedding layer to incorporate domain-specific information, which is added to the latent vector. This is consistent with your approach of using multiple decoders for different domains.
- Classifier: The `Classifier` class predicts the class of the input image from the latent representation. This is integrated at the bottleneck, as you described, to classify images based on their latent features.

### Step 4: Training and Loss Functions

- **Reparameterization Trick**: The `reparameterize()` function is correctly implemented to sample from the latent distribution using the mean and log variance, which is a standard technique in VAEs.
- **Loss Functions**:
    - The `vae_loss()` function combines reconstruction loss (MSE) and KL divergence, which is typical for VAEs.
    - The `compute_loss()` function calculates the total loss, including reconstruction, classification, and KL divergence losses, with adjustable weights (`alpha`, `beta`, `gamma`). This aligns with your approach of balancing different loss components.

### Step 5: Training Loop - **Training Function (`train_model`)**

- The function iterates over epochs and batches, performing forward and backward passes. It computes the loss using the `compute_loss()` function and updates model parameters using an optimizer.
- The model is trained on a source domain, and decoders are used to reconstruct images for target domains, which matches your strategy of using multiple decoders for domain generalization.

### Step 6: Evaluation - **Evaluation Function (`evaluate_model`)**

This function evaluates the model's performance on target domains by calculating classification accuracy and reconstruction loss. It uses the trained encoder, classifier, and the appropriate decoder for each target domain.