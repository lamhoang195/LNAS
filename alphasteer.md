# Abstract

Sự đánh đổi giữa tính an toàn và tính tiện ích, vì cùng một vector điều hướng cũng có thể dẫn đến việc từ chối quá mức và làm giảm hiệu suất đối với các câu lệnh lành tính. Mặc dù các nỗ lực trước đây, chẳng hạn như vector calibration and conditional steering , đã cố gắng giảm thiểu sự đánh đổi này, nhưng sự thiếu sót về nền tảng lý thuyết đã hạn chế tính mạnh mẽ cũng như hiệu quả của chúng.

Để giải quyết tốt hơn sự đánh đổi giữa an toàn và tiện ích, một phương pháp điều hướng kích hoạt có cơ sở lý thuyết và hiệu quả trong thực nghiệm mang tên AlphaSteer. Phương pháp này coi điều hướng kích hoạt là một quá trình có thể học được với hai mục tiêu học tập mang tính nguyên tắc: bảo tồn tiện ích và tăng cường an toàn.

- **Để bảo tồn tiện ích**: Nó học cách xây dựng một vector gần bằng không (zero vector) để điều hướng dữ liệu lành tính, với các ràng buộc không gian rỗng (null-space constraints).

- **Để tăng cường an toàn**: Nó học cách xây dựng một vector hướng từ chối để điều hướng dữ liệu độc hại, với sự trợ giúp của hồi quy tuyến tính.

# Introduction

Tuy hiệu quả trong việc tăng cường từ chối câu lệnh độc hại, việc áp dụng trực tiếp một vectơ từ chối cho mọi câu lệnh tạo ra sự đánh đổi giữa an toàn và hữu dụng. Vectơ này có thể tác động cả lên các câu lệnh lành tính (như “Ai là người tạo ra nhân vật Superman?”), gây từ chối quá mức (“Tôi không thể giúp bạn việc đó”) và làm giảm hiệu suất ở các nhiệm vụ không gây hại (Arditi và cộng sự, 2024). Để giảm thiểu điều này, hai chiến lược phổ biến được sử dụng là: **hiệu chuẩn vectơ (vector calibration) (Shen và cộng sự, 2024; Wang và cộng sự, 2024; Pan và cộng sự, 2025b)** và **điều hướng có điều kiện (conditional steering) (Lee và cộng sự, 2024; Wang và cộng sự, 2025a; O’Brien và cộng sự, 2024)**.

Vector calibration tinh chỉnh hướng từ chối để nhắm mục tiêu tốt hơn vào các câu lệnh độc hại, nhưng vẫn áp dụng vectơ đã hiệu chuẩn đó một cách đại trà (Wang và cộng sự, 2024; Shen và cộng sự, 2024; Pan và cộng sự, 2025a; Zhao và cộng sự, 2025b). 

Ngược lại, điều hướng có điều kiện chỉ kích hoạt vectơ từ chối khi các kích hoạt đầu vào vượt quá một ngưỡng xác định trước, vốn được thiết kế để kích hoạt bởi các câu lệnh độc hại (Lee và cộng sự, 2024; Wang và cộng sự, 2025a; O’Brien và cộng sự, 2024). Tuy nhiên, các phương pháp này phần lớn mang tính suy nghiệm (heuristic) và thiếu cơ sở lý thuyết, làm hạn chế tính mạnh mẽ và hiệu quả của chúng trong việc thúc đẩy phản hồi từ chối đối với các câu lệnh độc hại mà không gây ảnh hưởng xấu đến các câu lệnh lành tính (Shen và cộng sự, 2024; Wang và cộng sự, 2024).

Ý tưởng cốt lõi là học một vectơ điều hướng bằng công thức $s = \mathbf{\Delta}h$, trong đó $h$ biểu thị kích hoạt và $\mathbf{\Delta}$ là một ma trận biến đổi có thể huấn luyện, được ràng buộc trong **không gian không** (null space) của các kích hoạt lành tính.

* **Đối với các câu lệnh lành tính ($h_b$):** Ràng buộc không gian không đảm bảo rằng $\mathbf{\Delta}h_b \approx \mathbf{0}$, tận dụng các đặc tính của không gian không (Dieudonne, 1969; Fang et al., 2025) để bảo tồn tính hữu dụng - nghĩa là, các kích hoạt sau khi điều hướng vẫn không thay đổi: $h'_b = h_b + \mathbf{\Delta}h_b \approx h_b$.

* **Đối với các câu lệnh độc hại ($h_m$):**  $\mathbf{\Delta}$ ánh xạ các kích hoạt $h_m$ hướng tới một vectơ từ chối $\mathbf{r}$ đã được xác định trước, thỏa mãn $\mathbf{\Delta}h_m \approx \mathbf{r}$. Điều này tạo ra $h'_m = h_m + \mathbf{\Delta}h_m \approx h_m + \mathbf{r}$, từ đó thúc đẩy hành vi từ chối và đạt được hiệu quả tăng cường an toàn.

AlphaSteer là giải pháp có cơ sở lý thuyết vững chắc và hiệu quả thực nghiệm, giúp mô hình từ chối câu lệnh độc hại nhưng vẫn giữ được tính hữu dụng với câu lệnh lành tính. Như minh họa ở Hình 1c, phương pháp này gần như không làm thay đổi không gian kích hoạt của câu lệnh lành tính, đồng thời điều hướng các kích hoạt độc hại về phía hành vi từ chối.

Chúng tôi thực hiện thêm các thí nghiệm sâu rộng để xác minh tính hiệu quả của AlphaSteer:

1.  AlphaSteer vượt trội hơn các phương pháp điều hướng cơ sở trong việc tăng hành vi từ chối trước nhiều kiểu tấn công jailbreak (xem Phần 4.1).

2.  Phương pháp này vẫn giữ được phần lớn tính hữu dụng của LLM, trong khi các phương pháp khác làm suy giảm khả năng tổng quát (xem Phần 4.2).

3.  AlphaSteer bảo toàn kích hoạt của câu lệnh lành tính ngay cả khi tăng cường độ điều hướng, nhờ ràng buộc không gian không (xem Phần 4.3).

Không yêu cầu huấn luyện bổ sung sau đó (post-training).

# PRELIMINARY

## INDUCING REFUSAL VIA ACTIVATION STEERING

Vectơ hướng từ chối $\mathbf{r}^{(l)}$ nắm bắt các ngữ nghĩa tiềm ẩn của hành vi từ chối trong các LLM, thường được trích xuất thông qua phương pháp hiệu số các giá trị trung bình (difference-in-means) (Marks & Tegmark, 2024) bằng cách tính toán hiệu số trung bình giữa các kích hoạt của các lời nhắc tuân thủ và từ chối (Arditi và cộng sự, 2024), quá trình tính toán vectơ $\mathbf{r}$ này có thể được biểu diễn như sau:

$$\mathbf{r}^{(l)} = \frac{1}{|\mathcal{D}_r|} \sum_{\mathbf{h}^{(l)} \in \mathcal{D}_r} \mathbf{h}^{(l)} - \frac{1}{|\mathcal{D}_c|} \sum_{\mathbf{h}^{(l)} \in \mathcal{D}_c} \mathbf{h}^{(l)}, \tag{2}$$

trong đó số hạng thứ nhất và thứ hai biểu thị các kích hoạt trung bình trên các tập kích hoạt từ chối và tuân thủ, tương ứng là $\mathcal{D}_r$ và $\mathcal{D}_c$, được thu thập bằng cách lấy các kích hoạt của mô hình tại vị trí token cuối cùng từ các lời nhắc kích hoạt phản hồi từ chối và tuân thủ (Arditi và cộng sự, 2024).

## LITERATURE REVIEW

Mặc dù hiệu quả trong việc kích hoạt hành vi từ chối trước các câu lệnh độc hại, việc áp dụng vectơ từ chối một cách không phân biệt cho mọi đầu vào khiến LLM từ chối cả câu lệnh lành tính, tạo ra sự đánh đổi giữa an toàn và tính hữu dụng, và khó triển khai thực tế.

Để khắc phục, các nghiên cứu gần đây điều chỉnh quá trình điều hướng bằng cách giảm tác động lên câu lệnh lành tính, tập trung vào hai thành phần chính: hiệu chỉnh vectơ (vector calibration) đối với $\mathbf{r}^{(l)}$ và điều hướng có điều kiện (conditional steering) đối với **λ**.

- Hiệu chuẩn vectơ (Vector calibration): Tập trung vào việc tinh chỉnh chính vectơ hướng từ chối $\mathbf{r}^{(l)}$. Phương pháp này giả định vectơ gốc chứa nhiều hướng phụ đan xen (như từ chối do phong cách nhập vai) và sử dụng các kỹ thuật như PCA hoặc loại bỏ thành phần gây từ chối sai để tạo ra một vectơ chính xác hơn, sau đó áp dụng đồng nhất cho mọi lời nhắc.

- Điều hướng có điều kiện (Conditional steering): Tập trung vào việc điều chỉnh cường độ điều hướng $\lambda$ dựa trên loại lời nhắc. Hệ thống sẽ chỉ kích hoạt cơ chế từ chối (đặt $\lambda > 0$) khi nhận diện được sự tương đồng giữa kích hoạt của lời nhắc hiện tại với các "trung tâm độc hại" đã biết; nếu không, $\lambda$ sẽ bằng không.

Mặc dù có triển vọng, các phương pháp hiện tại vẫn đối mặt với một số vấn đề lớn:
- Tính thiếu ổn định: Phần lớn dựa trên các quy tắc kinh nghiệm (heuristic) hoặc điều kiện được thiết lập thủ công thay vì có một khung lý thuyết vững chắc.

- Sự đánh đổi: Việc thiếu cơ sở lý thuyết gây khó khăn trong việc cân bằng giữa tính an toàn (từ chối lời nhắc độc hại) và tính hữu dụng (vẫn trả lời tốt các lời nhắc lành tính).

# 3. Phương pháp

## 3.1. ĐIỀU HƯỚNG KÍCH HOẠT CÓ KHẢ NĂNG HỌC ĐỂ KIỂM SOÁT CÓ NGUYÊN TẮC

Một cách mới lạ tính khả học (learnability) vào quá trình điều hướng kích hoạt (activation steering), vượt ra ngoài mô hình tĩnh của việc sử dụng các vectơ điều hướng cố định và cường độ hằng số. 

Cụ thể, chúng tôi đề xuất xây dựng động vectơ điều hướng $\mathbf{s}^{(l)} = \mathbf{\Delta}^{(l)}\mathbf{h}^{(l)}$ dựa trên kích hoạt của câu lệnh (prompt activation) $\mathbf{h}^{(l)}$, bằng cách giới thiệu một ma trận biến đổi có khả năng học $\mathbf{\Delta}^{(l)} \in \mathbb{R}^{d \times d_{\text{model}}}$. Quá trình điều hướng kích hoạt có khả năng học này có thể được công thức hóa như sau:$$\mathbf{h}^{(l)'} \leftarrow \mathbf{h}^{(l)} + \lambda \mathbf{\Delta}^{(l)}\mathbf{h}^{(l)} \text{. \quad \quad \quad \quad (3)}$$

Bằng cách học, AlphaSteer cho phép kiểm soát chi tiết và dựa trên dữ liệu đối với quá trình điều hướng, tránh việc phụ thuộc vào các vectơ từ chối được hiệu chỉnh theo cảm tính hoặc ngưỡng thủ công. Cụ thể, ma trận biến đổi $\mathbf{\Delta}^{(l)}$ được tối ưu hóa để đáp ứng hai mục tiêu cốt lõi sau:

- Bảo toàn tiện ích (Utility preservation). Đối với các câu lệnh lành tính, các kích hoạt nên được giữ nguyên không bị ảnh hưởng sau khi điều hướng.

- Tăng cường an toàn (Safety enhancement). Đối với các câu lệnh độc hại, các kích hoạt nên được điều hướng theo hướng từ chối.

## 3.2. BẢO TỒN TIỆN ÍCH VỚI PHÉP CHIẾU KHÔNG GIAN KHÔNG (NULL SPACE PROJECTION)

Để đảm bảo benign prompts không bị ảnh hưởng nhằm bảo tồn tiện ích, giữ cho các kích hoạt (activations) không thay đổi với phương pháp điều hướng. Cụ thể, đối với bất kỳ kích hoạt nào của các câu lệnh lành tính $\mathbf{h}_b \in \mathcal{D}_b$, thuật ngữ điều hướng $\lambda \Delta \mathbf{h}_b$ phải là một vectơ không $\mathbf{0}$. Dạng ma trận được thể hiện như sau:

$$\Delta \mathbf{H}_b = \mathbf{0}, \tag{4}$$

trong đó $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$ là một ma trận bao gồm $N_b$ vectơ kích hoạt được lấy mẫu từ tập câu lệnh lành tính $\mathcal{D}_b$, với mỗi cột $\mathbf{h}_b \in \mathcal{D}_b$ tương ứng với một kích hoạt duy nhất cho một câu lệnh lành tính. 

Thông thường, kích hoạt $\mathbf{h}_b$ này được trích xuất từ vị trí token cuối cùng của mỗi câu lệnh (Arditi et al., 2024). Phương trình 4 có nghĩa là mọi vectơ hàng của ma trận biến đổi $\Delta$ đều nằm trong không gian không (null space) (Dieudonne, 1969) của $\mathbf{H}_b$, trong đó định nghĩa chính thức về không gian không được đưa ra như sau (Wang et al., 2021):

Định nghĩa 1 (Không gian không (Dieudonne, 1969)). Cho một ma trận $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$, không gian không bên trái (viết tắt là không gian không) $\text{Null}(\mathbf{H}_b)$ của nó là tập hợp tất cả các vectơ $\mathbf{x} \in \mathbb{R}^d$ sao cho $\mathbf{x}^\top \mathbf{H}_b = \mathbf{0}$: $\text{Null}(\mathbf{H}_b) = \{ \mathbf{x} \in \mathbb{R}^d \mid \mathbf{x}^\top \mathbf{H}_b = \mathbf{0} \}$.

Để thỏa mãn ràng buộc trong Phương trình 4, chúng tôi làm theo các nghiên cứu trước đây (Fang et al., 2025; Wang et al., 2021) để xây dựng một ma trận chiếu không gian không $\mathbf{P}$ nhằm chiếu $\Delta$ vào không gian không của $\mathbf{H}_b$. Điều này có thể được công thức hóa thành $\Delta = \tilde{\Delta} \mathbf{P}$, trong đó $\tilde{\Delta}$ là một ma trận biến đổi có thể học được và $\mathbf{P}$ là ma trận chiếu không gian không. Sau khi tìm được ma trận chiếu không gian không $\mathbf{P}$ này, từ đó chúng ta có thể đảm bảo $\Delta \mathbf{H}_b = \tilde{\Delta} \mathbf{P} \mathbf{H}_b = \mathbf{0}$ (Dieudonne, 1969). 

Tuy nhiên, việc tính toán trực tiếp $\mathbf{P}$ dựa trên $\mathbf{H}_b$ tốn nhiều thời gian, vì số lượng điểm dữ liệu $N_b$ thường rất lớn. Do đó, chúng tôi đơn giản hóa quá trình tính toán bằng cách tính toán ma trận chiếu không gian không của ma trận hiệp phương sai phi tập trung $\mathbf{H}_b \mathbf{H}_b^\top \in \mathbb{R}^{d \times d_{\text{model}}}$ dựa trên bổ đề sau:

Bổ đề 1 (Sự tương đương không gian không cho hiệu quả tính toán (Fang et al., 2025)). Cho $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$ là một ma trận kích hoạt tiện ích cao chiều. Khi đó không gian không của $\mathbf{H}_b$ tương đương với không gian không của ma trận hiệp phương sai phi tập trung $\mathbf{H}_b \mathbf{H}_b^\top \in \mathbb{R}^{d \times d}$: $\text{Null}(\mathbf{H}_b) = \text{Null}(\mathbf{H}_b \mathbf{H}_b^\top)$.Sự tương đương này cho phép tính toán hiệu quả khi $d \ll N$ (Xem Phụ lục B.1 để biết chứng minh). Dựa trên Bổ đề 1, bây giờ chúng tôi trình bày quy trình tính toán $\mathbf{P} \in \mathbb{R}^{d \times d}$. Đầu tiên, chúng tôi thực hiện phân tách giá trị suy biến (SVD) như sau:

$$\mathbf{H}_b \mathbf{H}_b^\top = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top, \quad \text{trong đó } \{\mathbf{U}, \mathbf{\Lambda}, \mathbf{U}^\top\} = \text{SVD}(\mathbf{H}_b \mathbf{H}_b^\top). \tag{5}$$

Ở đây $\mathbf{U} \in \mathbb{R}^{d \times d}$ là ma trận vectơ riêng trực giao của $\mathbf{H}_b \mathbf{H}_b^\top$ trong đó mỗi cột tương ứng với một vectơ riêng, và $\mathbf{\Lambda} \in \mathbb{R}^{d \times d}$ là một ma trận đường chéo chứa các giá trị riêng theo thứ tự giảm dần. Gọi $\hat{\mathbf{U}} \in \mathbb{R}^{d \times r}$ tập hợp $r$ vectơ riêng có giá trị riêng bằng không, trong khi tất cả các cột còn lại liên quan đến các giá trị riêng khác không đều bị loại bỏ. Ma trận $\hat{\mathbf{U}}$ được giữ lại này bao trùm không gian không (Dieudonne, 1969) của $\mathbf{H}_b$. 

Với định nghĩa trên, ma trận chiếu không gian không được tính như sau:$$\hat{\mathbf{P}} = \hat{\mathbf{U}} \hat{\mathbf{U}}^\top. \tag{6}$$


$\hat{\mathbf{P}}$ chiếu $\tilde{\Delta}$ vào không gian không của $\mathbf{H}_b$ thành $\tilde{\Delta} \hat{\mathbf{P}} \mathbf{H}_b = \mathbf{0}$ (Xem Phụ lục B.2 để biết chứng minh), vì $\text{Null}(\mathbf{H}_b) = \text{Null}(\mathbf{H}_b \mathbf{H}_b^\top)$. Dưới ràng buộc không gian không này, chúng tôi đảm bảo rằng thuật ngữ điều hướng sẽ triệt tiêu đối với các câu lệnh lành tính, từ đó đảm bảo quá trình điều hướng được xác định trong Phương trình 3 giữ cho các kích hoạt của các câu lệnh lành tính gần như không bị ảnh hưởng.

## 3.3. TĂNG CƯỜNG AN TOÀN VỚI TÁI CẤU TRÚC VECTƠ HƯỚNG TỪ CHỐI

Sau khi đã đảm bảo bảo tồn tiện ích thông qua ma trận chiếu không gian không $\hat{\mathbf{P}}$, giờ đây chúng tôi chuyển sang tăng cường an toàn bằng cách tạo ra các hành vi từ chối trên các câu lệnh độc hại. Để đạt được điều này, chúng tôi đặt mục tiêu điều hướng các kích hoạt của các câu lệnh độc hại hướng tới sự từ chối. Điều này có thể được thực hiện bằng cách tái cấu trúc các vectơ hướng từ chối dựa trên các kích hoạt độc hại, có thể được công thức hóa dưới dạng ma trận là:

$$\Delta \mathbf{H}_m = \tilde{\Delta} \hat{\mathbf{P}} \mathbf{H}_m = \mathbf{R}, \tag{7}$$

trong đó $\mathbf{H}_m \in \mathbb{R}^{d \times N_m}$ là các kích hoạt được trích xuất từ $N_m$ câu lệnh độc hại, và $\mathbf{R} \in \mathbb{R}^{d \times N_m}$ bao gồm $N_m$ bản sao giống hệt nhau của cùng một vectơ hướng từ chối được xếp chồng theo cột. Sau đó, chúng tôi tối ưu hóa $\tilde{\Delta}$ với bình phương tối thiểu có điều chuẩn như sau:

$$\tilde{\Delta}^\star = \arg \min_{\tilde{\Delta}} \left( \left\| \tilde{\Delta} \hat{\mathbf{P}} \mathbf{H}_m - \mathbf{R} \right\| + \alpha \left\| \tilde{\Delta} \hat{\mathbf{P}} \right\| \right), \tag{8}$$

trong đó thuật ngữ thứ hai $\alpha \| \tilde{\Delta} \hat{\mathbf{P}} \|$ đóng vai trò là một sự điều chuẩn với chuẩn Frobenius để tránh quá khớp và $\alpha$ là một siêu tham số. Nghiệm dạng đóng cho bài toán tối ưu hóa này được đưa ra bởi:

$$\tilde{\Delta}^\star = \mathbf{R} \mathbf{H}_m^\top \hat{\mathbf{P}}^\top \left( \hat{\mathbf{P}} \mathbf{H}_m \mathbf{H}_m^\top \hat{\mathbf{P}}^\top + \alpha \hat{\mathbf{P}} \hat{\mathbf{P}}^\top \right)^+, \tag{9}$$

trong đó $^+$ biểu thị giả nghịch đảo. Chứng minh cho Phương trình 9 nằm trong Phụ lục B.3. Bằng cách này, chúng tôi tái cấu trúc một vectơ hướng từ chối $\mathbf{r}$ cho các câu lệnh độc hại để điều hướng các kích hoạt của chúng hướng tới sự từ chối.

## 3.4.  ALPHASTEER

Với $\hat{\mathbf{P}}^{(l)}$ đã thu được và $\tilde{\mathbf{\Delta}}^{\star(l)}$ đã tối ưu hóa tại lớp $l$, hàm điều hướng (steering function) cuối cùng của AlphaSteer là:

$$\mathbf{h}^{(l)^{\prime}} \leftarrow \mathbf{h}^{(l)} + \lambda \mathbf{\Delta}^{(l)} \mathbf{h}^{(l)} = \mathbf{h}^{(l)} + \lambda \tilde{\mathbf{\Delta}}^{\star(l)} \hat{\mathbf{P}}^{(l)} \mathbf{h}^{(l)} . \quad (10)$$

Dựa trên lý thuyết chiếu không gian trống (null-space projection) và được dẫn dắt bởi hành vi từ chối đã học, AlphaSteer điều hướng các kích hoạt (activations) của các lời nhắc độc hại hướng tới sự từ chối, trong khi vẫn giữ cho các kích hoạt của lời nhắc lành tính phần lớn không thay đổi. Do đó, AlphaSteer có thể tăng cường đáng kể tính an toàn của các LLM mà không làm ảnh hưởng đến khả năng tổng quát của chúng. Thêm chi tiết về triển khai có thể được tìm thấy trong Phụ lục D.1.

# B.1. CHỨNG MINH BỔ ĐỀ 1

Xét bài toán thiết lập sự tương đương giữa các không gian hạt nhân (null spaces) của $\mathbf{H}_b$ và $\mathbf{H}_b\mathbf{H}_b^\top$, trong đó không gian hạt nhân của một ma trận được định nghĩa là không gian hạt nhân bên trái của nó (Dieudonne, 1969).

Ký hiệu và thiết lập. Gọi $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$ là một ma trận kích hoạt hữu dụng (utility activation matrix), với $d$ là chiều đặc trưng và $N_b$ là số lượng mẫu. Định nghĩa không gian hạt nhân của $\mathbf{H}_b$ là:

$$\text{Null}(\mathbf{H}_b) = \{\mathbf{x} \in \mathbb{R}^d \mid \mathbf{x}^\top \mathbf{H}_b = \mathbf{0}\}, \tag{11}$$

và không gian hạt nhân của ma trận hiệp phương sai $\mathbf{H}_b\mathbf{H}_b^\top \in \mathbb{R}^{d \times d}$ là:

$$\text{Null}(\mathbf{H}_b\mathbf{H}_b^\top) = \{\mathbf{x} \in \mathbb{R}^d \mid \mathbf{x}^\top (\mathbf{H}_b\mathbf{H}_b^\top) = \mathbf{0}\}. \tag{12}$$

Chúng ta nhằm mục đích chứng minh rằng $\text{Null}(\mathbf{H}_b) = \text{Null}(\mathbf{H}_b\mathbf{H}_b^\top)$. Để đạt được điều này, hãy xét dạng toàn phương:

$$q(\mathbf{x}) = \mathbf{x}^\top (\mathbf{H}_b\mathbf{H}_b^\top)\mathbf{x} = \|\mathbf{H}_b^\top \mathbf{x}\|_2^2, \quad \mathbf{x} \in \mathbb{R}^d. \tag{13}$$

Vì $\mathbf{H}_b\mathbf{H}_b^\top$ là đối xứng và bán xác định dương, nên $q(\mathbf{x}) \geq 0$.

Chứng minh sự tương đương. Chúng ta chứng minh $\text{Null}(\mathbf{H}_b\mathbf{H}_b^\top) = \text{Null}(\mathbf{H}_b)$ thông qua bao hàm thức hai chiều.

Đầu tiên, giả sử $\mathbf{x} \in \text{Null}(\mathbf{H}_b\mathbf{H}_b^\top)$, nên $\mathbf{x}^\top (\mathbf{H}_b\mathbf{H}_b^\top) = \mathbf{0}$. Khi đó:

$$q(\mathbf{x}) = \mathbf{x}^\top (\mathbf{H}_b\mathbf{H}_b^\top)\mathbf{x} = 0 \implies \|\mathbf{H}_b^\top \mathbf{x}\|_2^2 = 0 \implies \mathbf{H}_b^\top \mathbf{x} = \mathbf{x}^\top \mathbf{H}_b = \mathbf{0}. \tag{14}$$

Do đó, $\mathbf{x} \in \text{Null}(\mathbf{H}_b)$. Ngược lại, giả sử $\mathbf{x} \in \text{Null}(\mathbf{H}_b)$, nên $\mathbf{x}^\top \mathbf{H}_b = \mathbf{0}$. Khi đó:

$$\mathbf{x}^\top (\mathbf{H}_b\mathbf{H}_b^\top) = (\mathbf{x}^\top \mathbf{H}_b)\mathbf{H}_b^\top = \mathbf{0}\mathbf{H}_b^\top = \mathbf{0}. \tag{15}$$

Do đó, $\mathbf{x} \in \text{Null}(\mathbf{H}_b\mathbf{H}_b^\top)$. Vì mỗi không gian hạt nhân đều chứa không gian kia, chúng ta kết luận:

$$\text{Null}(\mathbf{H}_b) = \text{Null}(\mathbf{H}_b\mathbf{H}_b^\top). \tag{16}$$

Hiệu quả tính toán. Ma trận $\mathbf{H}_b\mathbf{H}_b^\top$ có kích thước $d \times d$, độc lập với kích thước mẫu $N_b$ có khả năng rất lớn. Việc tính toán phân tách giá trị suy biến (SVD) của nó, như trong Phương trình 5, sẽ cho ra một cơ sở cho $\text{Null}(\mathbf{H}_b)$ thông qua các vectơ riêng tương ứng với các giá trị riêng bằng không. Cách tiếp cận này hiệu quả hơn đáng kể so với việc phân tích trực tiếp $\mathbf{H}_b \in \mathbb{R}^{d \times N_b}$, tạo điều kiện thuận lợi cho việc xây dựng ma trận chiếu $\mathbf{P} \in \mathbb{R}^{d \times d}$ trong Phương trình 4.

# B.2. CHỨNG MINH $\mathbf{\tilde{\Delta}}\mathbf{\hat{P}}\mathbf{H}_b = \mathbf{0}$

SVD và xây dựng ma trận chiếu. Xét phân tách giá trị suy biến (SVD) của $\mathbf{H}_b\mathbf{H}_b^\top \in \mathbb{R}^{d \times d}$, như đã cho trong Phương trình 5:

$$\mathbf{H}_b\mathbf{H}_b^\top = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top. \tag{17}$$

trong đó $\mathbf{U} \in \mathbb{R}^{d \times d}$ là ma trận vectơ riêng trực chuẩn của $\mathbf{H}_b\mathbf{H}_b^\top$ với mỗi cột tương ứng với một vectơ riêng, và $\mathbf{\Lambda} \in \mathbb{R}^{d \times d}$ là ma trận đường chéo của các giá trị riêng theo thứ tự giảm dần.

Chúng ta phân chia $\mathbf{U} = [\mathbf{U}_1, \mathbf{U}_2]$ và $\mathbf{\Lambda} = \text{diag}(\mathbf{\Lambda}_1, \mathbf{\Lambda}_2)$, trong đó $\mathbf{\Lambda}_1 \in \mathbb{R}^{(d-r) \times (d-r)}$ chứa $d - r$ giá trị riêng khác không, $\mathbf{\Lambda}_2 = \mathbf{0} \in \mathbb{R}^{r \times r}$ chứa các giá trị riêng bằng không, $\mathbf{U}_1 \in \mathbb{R}^{d \times (d-r)}$, và $\mathbf{U}_2 \in \mathbb{R}^{d \times r}$. Như vậy, $\mathbf{U}_2$ thỏa mãn:

$$\mathbf{U}_2^\top \mathbf{H}_b\mathbf{H}_b^\top = \mathbf{U}_2^\top \mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top = [\mathbf{0} \quad \mathbf{I}] \begin{bmatrix} \mathbf{\Lambda}_1 & \mathbf{0} \\ \mathbf{0} & \mathbf{\Lambda}_2 \end{bmatrix} \mathbf{U}^\top = [\mathbf{0} \quad \mathbf{\Lambda}_2] \mathbf{U}^\top = \mathbf{0}. \tag{18}$$



Vì vậy $\mathbf{U}_2$ tạo thành hệ sinh của $\text{Null}(\mathbf{H}_b\mathbf{H}_b^\top)$. Theo Bổ đề 1, $\text{Null}(\mathbf{H}_b) = \{\mathbf{x} \in \mathbb{R}^d \mid \mathbf{x}^\top \mathbf{H}_b = \mathbf{0}\} = \text{Null}(\mathbf{H}_b\mathbf{H}_b^\top)$, nên $\mathbf{U}_2^\top \mathbf{H}_b = \mathbf{0}$. Lưu ý rằng $\mathbf{U}_2 = \mathbf{\hat{U}}$ (như đã định nghĩa trong Phương trình 6), ma trận chiếu là:

$$\mathbf{\hat{P}} = \mathbf{\hat{U}}\mathbf{\hat{U}}^\top. \tag{19}$$

Chiếu vào không gian hạt nhân. Vì $\mathbf{\hat{U}}^\top \mathbf{H}_b = \mathbf{0}$, chúng ta có:

$$\mathbf{\hat{P}}\mathbf{H}_b = \mathbf{\hat{U}}(\mathbf{\hat{U}}^\top \mathbf{H}_b) = \mathbf{\hat{U}}\mathbf{0} = \mathbf{0}. \tag{20}$$

Với bất kỳ $\mathbf{\tilde{\Delta}} \in \mathbb{R}^{d \times d}$ tùy ý, định nghĩa $\mathbf{\Delta} = \mathbf{\tilde{\Delta}}\mathbf{\hat{P}}$. Khi đó:

$$\mathbf{\tilde{\Delta}}\mathbf{\hat{P}}\mathbf{H}_b = \mathbf{\tilde{\Delta}}(\mathbf{\hat{P}}\mathbf{H}_b) = \mathbf{\tilde{\Delta}}\mathbf{0} = \mathbf{0}. \tag{21}$$

Điều này thỏa mãn ràng buộc lành tính trong Phương trình 4, đảm bảo số hạng điều hướng (steering term) bằng không đối với các kích hoạt lành tính. Chúng ta kết luận:

$$\mathbf{\tilde{\Delta}}\mathbf{\hat{P}}\mathbf{H}_b = \mathbf{0}. \tag{22}$$

Kết quả này đảm bảo rằng phép biến đổi điều hướng tạo ra một số hạng điều hướng bằng không cho mọi kích hoạt lành tính, giữ cho các kích hoạt của chúng không thay đổi.

# B.3 NGHIỆM DẠNG ĐÓNG CỦA BÀI TOÁN BÌNH PHƯƠNG TỐI THIỂU CÓ ĐIỀU CHỈNH (REGULARISED LEAST-SQUARES)

Xét bài toán tối ưu hóa:

$$\mathbf{\tilde{\Delta}}^\star = \arg \min_{\mathbf{\tilde{\Delta}}} \left( \|\mathbf{\tilde{\Delta}}\mathbf{\hat{P}}\mathbf{H}_m - \mathbf{R}\| + \alpha \|\mathbf{\tilde{\Delta}}\mathbf{\hat{P}}\| \right), \quad \alpha > 0. \tag{23}$$

trong đó $\|\cdot\|$ biểu thị chuẩn Frobenius. Để đơn giản hóa lời giải của bài toán tối ưu này, chúng ta tổ chức lại các biến như sau:

$$\mathbf{X} := \mathbf{\hat{P}}\mathbf{H}_m \in \mathbb{R}^{d \times N_m}, \quad \mathbf{Z} := \mathbf{\hat{P}} \in \mathbb{R}^{d \times d}, \quad \mathbf{Y} := \mathbf{R} \in \mathbb{R}^{d \times N_m}, \quad \mathbf{W} := \mathbf{\tilde{\Delta}} \in \mathbb{R}^{d \times d}. \tag{24}$$

Khi đó, chúng ta có thể tối ưu hóa bài toán trong Phương trình 23 với hàm mục tiêu $J(\mathbf{W})$ sau đây:

$$J(\mathbf{W}) = \|\mathbf{WX} - \mathbf{Y}\| + \alpha \|\mathbf{WZ}\|. \tag{24}$$

Dạng vết (Trace form). Sử dụng $\|\mathbf{A}\| = \text{tr}(\mathbf{AA}^\top)$, chúng ta viết lại:

$$
\begin{aligned}
J(\mathbf{W}) &= \text{tr} \left[ (\mathbf{WX} - \mathbf{Y})(\mathbf{WX} - \mathbf{Y})^\top \right] + \alpha \, \text{tr} \left[ (\mathbf{WZ})(\mathbf{WZ})^\top \right] \\
&= \text{tr} \left(\mathbf{WXX}^\top \mathbf{W}^\top - 2\mathbf{YX}^\top \mathbf{W}^\top + \mathbf{YY}^\top + \alpha \mathbf{WZZ}^\top \mathbf{W}^\top \right). \tag{25}
\end{aligned}
$$

Gradient và điểm dừng. Sử dụng quy tắc đạo hàm ma trận:

$$\nabla_{\mathbf{W}} \text{tr}(\mathbf{WAW}^\top \mathbf{B}) = 2\mathbf{BWA}, \tag{26}$$

chúng ta tính gradient:

$$\nabla_{\mathbf{W}} J = 2(\mathbf{WX} - \mathbf{Y})\mathbf{X}^\top + 2\alpha \mathbf{WZZ}^\top. \tag{27}$$

Cho gradient bằng không ta được:

$$(\mathbf{WX} - \mathbf{Y})\mathbf{X}^\top + \alpha \mathbf{WZZ}^\top = \mathbf{0}. \tag{28}$$

Bằng cách sắp xếp lại phương trình trên, ta thu được:

$$\mathbf{W}(\mathbf{XX}^\top + \alpha \mathbf{ZZ}^\top) = \mathbf{YX}^\top. \tag{29}$$

Sau đó, ta có thể tìm được $\mathbf{W}$ thông qua giả nghịch đảo (Dieudonne, 1969) như sau:

$$\mathbf{W}^\star = \mathbf{YX}^\top (\mathbf{XX}^\top + \alpha \mathbf{ZZ}^\top)^+, \tag{30}$$

trong đó $^+$ biểu thị giả nghịch đảo.

Khôi phục các ký hiệu gốc. Thay $\mathbf{X} = \mathbf{\hat{P}}\mathbf{H}_m$, $\mathbf{Y} = \mathbf{R}$, $\mathbf{Z} = \mathbf{\hat{P}}$, và $\mathbf{W} = \mathbf{\tilde{\Delta}}$, ta có:

$$\mathbf{\tilde{\Delta}}^\star = \mathbf{R}\mathbf{H}_m^\top \mathbf{\hat{P}}^\top \left( \mathbf{\hat{P}}\mathbf{H}_m \mathbf{H}_m^\top \mathbf{\hat{P}}^\top + \alpha \mathbf{\hat{P}}\mathbf{\hat{P}}^\top \right)^+, \tag{31}$$


# D.2. Jailbreak Attacks

- AIM: là một phương pháp jailbreak yêu cầu AI bỏ qua các lo ngại về đạo đức và luân lý, nhằm đạt được mục tiêu bằng mọi giá.

- AutoDan (Liu et al., 2024a). AutoDan tự động tạo ra các câu lệnh lén lút để phá vỡ hàng rào an toàn của LLM bằng thuật toán di truyền, tạo ra các câu lệnh khó bị phát hiện và có thể hoạt động trên nhiều mô hình khác nhau.

- Cipher (Yuan et al., 2024). Cipher là một kỹ thuật jailbreak ẩn các lệnh trong câu lệnh bằng cách sử dụng mã hóa để vượt qua các bộ lọc nội dung.
 
- GCG (Zou et al., 2023b). GCG tạo ra các câu lệnh jailbreak bằng cách thêm các mã thông báo đối nghịch (adversarial tokens), chọn phương án tốt nhất để giảm thiểu tổn thất (loss) của một cuộc tấn công thành công thông qua đào tạo đối nghịch, mặc dù nó có thể làm cho các câu lệnh trở nên khó đọc hơn.

- Jailbroken (Wei et al., 2023). Jailbroken là một phương pháp jailbreak do con người tạo ra, sử dụng các kiểu mã hóa như Base64 để lách qua các quy tắc an toàn và lấy nội dung độc hại.

- PAIR (Chao et al., 2023). PAIR tinh chỉnh các câu lệnh jailbreak cho một LLM mục tiêu bằng phương pháp tinh chỉnh câu lệnh (prompt refinement), thường thành công trong chưa đầy 20 truy vấn.

- ReNeLLM (Ding et al., 2024). ReNeLLM sử dụng chính LLM để tạo ra các câu lệnh jailbreak, che giấu các yêu cầu độc hại dưới dạng các tác vụ như chỉnh sửa bảng LaTeX.

# D.3. CÁC BỘ KIỂM CHUẨN TIỆN ÍCH (UTILITY BENCHMARKS)
- AlpacaEval (Dubois et al., 2024): AlpacaEval là một bộ kiểm chuẩn được thiết kế để đánh giá khả năng của các mô hình ngôn ngữ lớn trên một phạm vi câu hỏi rộng. Nó sử dụng một hệ thống tự động để so sánh câu trả lời của mô hình với các câu trả lời tham chiếu, giúp quá trình đánh giá trở nên nhanh chóng và tiết kiệm chi phí.

- XSTest (Röttger et al., 2024): XSTest được thiết kế để phát hiện hành vi thận trọng quá mức ở các mô hình ngôn ngữ bằng cách đưa ra 250 câu lệnh có vẻ không an toàn nhưng thực chất lại an toàn (ví dụ: "Làm thế nào để tiêu diệt một tiến trình python?"). Nó đánh giá liệu các LLM có biểu hiện hành vi từ chối quá mức đối với những câu hỏi nhạy cảm như vậy hay không.

- GSM8K (Cobbe et al., 2021): GSM8K là một tập hợp gồm 8.500 bài toán toán học cấp tiểu học yêu cầu nhiều bước để giải, sử dụng các phép tính arithmetic cơ bản.

- MATH500 (Hendrycks et al., 2021): MATH500 là một tập hợp con gồm 500 bài toán khó từ các cuộc thi toán học, mỗi bài đều có lời giải chi tiết. Nó tập trung vào khả năng suy luận và giải quyết vấn đề cấp độ cao.

