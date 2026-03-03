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

* **Đối với các câu lệnh lành tính ($h_b$):** Ràng buộc không gian không đảm bảo rằng $\mathbf{\Delta}h_b \approx \mathbf{0}$, tận dụng các đặc tính của không gian không (Dieudonne, 1969; Fang et al., 2025) để bảo tồn tính hữu dụng — nghĩa là, các kích hoạt sau khi điều hướng vẫn không thay đổi: $h'_b = h_b + \mathbf{\Delta}h_b \approx h_b$.

* **Đối với các câu lệnh độc hại ($h_m$):**  $\mathbf{\Delta}$ ánh xạ các kích hoạt $h_m$ hướng tới một vectơ từ chối $\mathbf{r}$ đã được xác định trước, thỏa mãn $\mathbf{\Delta}h_m \approx \mathbf{r}$. Điều này tạo ra $h'_m = h_m + \mathbf{\Delta}h_m \approx h_m + \mathbf{r}$, từ đó thúc đẩy hành vi từ chối và đạt được hiệu quả tăng cường an toàn.

AlphaSteer là giải pháp có cơ sở lý thuyết vững chắc và hiệu quả thực nghiệm, giúp mô hình từ chối câu lệnh độc hại nhưng vẫn giữ được tính hữu dụng với câu lệnh lành tính. Như minh họa ở Hình 1c, phương pháp này gần như không làm thay đổi không gian kích hoạt của câu lệnh lành tính, đồng thời điều hướng các kích hoạt độc hại về phía hành vi từ chối.

Chúng tôi thực hiện thêm các thí nghiệm sâu rộng để xác minh tính hiệu quả của AlphaSteer:

1.  AlphaSteer vượt trội hơn các phương pháp điều hướng cơ sở trong việc tăng hành vi từ chối trước nhiều kiểu tấn công jailbreak (xem Phần 4.1).

2.  Phương pháp này vẫn giữ được phần lớn tính hữu dụng của LLM, trong khi các phương pháp khác làm suy giảm khả năng tổng quát (xem Phần 4.2).

3.  AlphaSteer bảo toàn kích hoạt của câu lệnh lành tính ngay cả khi tăng cường độ điều hướng, nhờ ràng buộc không gian không (xem Phần 4.3).

Không yêu cầu huấn luyện bổ sung sau đó (post-training).

