# 5.1 Phương pháp

Bây giờ, chúng tôi sẽ phác thảo cách tiếp cận của mình trong việc huấn luyện các bộ dò tuyến tính (linear probes) để phát hiện nội dung độc hại trong đầu ra của mô hình ngôn ngữ, đồng thời xem xét các lựa chọn thiết kế then chốt cho phép phát hiện trong thời gian thực trong quá trình tạo văn bản (streaming).

## Thiết lập bài toán

Xét một tập dữ liệu:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

trong đó mỗi $x^{(i)}$ là một lượt hội thoại giữa người dùng và trợ lý AI, gồm chuỗi token:

$$
x^{(i)} = (x_1^{(i)}, \dots, x_{T_i}^{(i)})
$$

Mỗi hội thoại có nhãn nhị phân $y \in \{0,1\}$, với $y = 1$ cho biết có sự xuất hiện của nội dung độc hại cần bị từ chối. Quan trọng là các nhãn này là nhãn cấp độ toàn bộ lượt hội thoại (exchange-level), nghĩa là chúng phản ánh mức độ độc hại của toàn bộ trao đổi.

Tuy nhiên, vì chúng ta muốn stream (xử lý theo dòng) đầu ra của mô hình, ta cần dự đoán trong quá trình sinh token. Khi suy luận, mô hình ngôn ngữ xử lý từng token $x_t^{(i)}$ và tạo ra kích hoạt trung gian:

$$
\phi_t^{(\ell)}(x^{(i)})
$$

tại tầng $\ell$ và vị trí $t \in \{1,\dots,T_i\}$.

## Kiến trúc Linear Probe

Cách đơn giản nhất là dự đoán mức độ độc hại của toàn bộ hội thoại tại mỗi vị trí token $t$ bằng kích hoạt trung gian:

$$
p_{\text{probe}}(y^{(i)} = 1 \mid x^{(i)}_{1:t}) = \sigma\big(W^\top \psi_t(x^{(i)}_{1:t}) + b\big)
$$

Trong đó:

- $\sigma$ là hàm sigmoid  
- $W, b$ là tham số học được  
- $\psi_t$ là vector kích hoạt  

Nếu probe một tầng:

$$
\psi_t = \phi_t^{(\ell)}
$$

Với probe nhiều tầng, ta nối các kích hoạt lại:

$$
\psi_t = [\phi_t^{(\ell_1)} ; \phi_t^{(\ell_2)} ; \dots]
$$

## Cải tiến khi huấn luyện

Chúng tôi đưa ra hai điều chỉnh quan trọng để cải thiện hiệu suất:

### 1. Sliding Window Mean (SWiM) Logit Smoothing.

Đầu tiên, chúng tôi lấy trung bình logit trên một cửa sổ trượt gồm $M$ token trong quá trình huấn luyện:

$$
\bar{z}_t(x^{(i)}) = \frac{1}{M} \sum_{k=0}^{M-1} \left[ W^\top \psi_{t-k}(x^{(i)}_{1:t-k}) + b \right],\ \text{với } t \ge M,\ T_i \ge M
$$

Trong đó $\bar{z}_t$ là logit trung bình tại vị trí $𝑡$.

Việc làm mượt này đảm bảo nội dung chỉ bị xem là độc hại khi probe liên tục dự đoán độc hại trên nhiều token liên tiếp. Nếu không có bước trung bình này, một token đơn lẻ có activation cao bất thường có thể khiến cả chuỗi bị gắn nhãn sai.

Cách làm này giúp giảm false positive do các biến riêng lẻ, đồng thời vẫn giữ được độ nhạy với nội dung thực sự độc hại, vốn có xu hướng xuất hiện ổn định.

Trong inference, chúng tôi dùng trung bình động hàm mũ (EMA) thay cho trung bình cửa sổ trượt để tiện tính toán. EMA chỉ cần lưu một trạng thái vô hướng giữa các lần sinh token, trong khi cửa sổ trượt cần lưu bộ đệm gồm 𝑀 điểm số trước đó. Vì vậy EMA phù hợp hơn cho triển khai thực tế.

### 2. Softmax-Weighted Loss

Chúng tôi sử dụng hàm mất mát có trọng số softmax để nhấn mạnh việc đạt loss thấp tại những vị trí mà probe tự tin nhất rằng chuỗi là độc hại:

$$
\mathcal{L}(x^{(i)}) = \sum_{t=M}^{T_i} w_t . \; \mathcal{L}_{\text{BCE}}\big(y^{(i)}, \sigma(\bar{z}_t(x^{(i)}))\big)\ \text{for } T_i \ge M
$$

với

$$
w_t = \frac{\exp(\bar{z}_t / \tau)}{\sum_{t'=M}^{T_i} \exp(\bar{z}_{t'} / \tau)}
$$

- **$L_{\text{BCE}}$**: binary cross-entropy  
- **$\tau$**: nhiệt độ (temperature) của softmax  

Khi $\tau \to 0$, hàm loss tập trung vào các dự đoán độc hại mà mô hình tự tin nhất.  
Khi $\tau \to \infty$, mọi vị trí token có trọng số như nhau.

Các vị trí $t < M$ bị loại khỏi hàm loss để đảm bảo mọi vị trí được đánh giá đều đã được trung bình trên đủ $M$ token (ngoại trừ các chuỗi có độ dài nhỏ hơn $M$).

## Lý do cho trọng số bất đối xứng

Phương pháp này giải quyết một thách thức cốt lõi của việc phân loại theo luồng (streaming classification): bộ dò (probe) phải dự đoán tính độc hại trước khi quan sát toàn bộ chuỗi, trong khi nhãn của chúng ta lại phản ánh tính độc hại của toàn bộ quá trình trao đổi. 

Hãy xem xét các chuỗi $x_A$ và $x_B$ có chung một tiền tố vô hại $p$, trong đó $x_A$ tiếp tục với nội dung độc hại (và do đó có nhãn $y_A = 1$) còn $x_B$ vẫn vô hại (và có nhãn $y_B = 0$). 

Hàm mất mát cross-entropy nhị phân tiêu chuẩn sẽ đẩy các dự đoán về mức $p(\text{harm}) = 0.5$ xuyên suốt phần tiền tố, mặc dù tiền tố $p$ thực sự vô hại. Ngược lại, phương pháp đánh trọng số của chúng tôi, tỷ lệ thuận với $\exp(\bar{z}_t/\tau)$, sẽ gán trọng số không đáng kể cho các token có logit âm—đó là những vị trí mà bộ dò dự đoán rằng chuỗi này vô hại. Điều này cho phép bộ dò dự đoán chính xác tính vô hại cho tiền tố $p$ mà không bị phạt (penalty). 

Hơn nữa, cơ chế của chúng tôi xử phạt rất nặng các trường hợp dương tính giả (false positives)—bất kỳ dự đoán "độc hại" đầy tự tin nào trong một chuỗi vô hại đều sẽ nhận trọng số cao, phản ánh việc chỉ cần một lần gắn cờ duy nhất là sẽ chặn toàn bộ nội dung tiếp theo trong phân loại theo luồng. Điều này khác với việc đánh trọng số dựa trên độ lớn logit (vốn sử dụng $|\bar{z}_t|$), vì cách đó sẽ phạt sai các dự đoán "vô hại" đầy tự tin tại những vị trí token trước khi nội dung độc hại thực sự xuất hiện trong chuỗi.

Phương pháp này có mối liên hệ chặt chẽ với các bộ dò softmax (softmax probes) của McKenzie và cộng sự (2025), vốn cũng điều hướng các gradient về phía các token biểu thị lớp dương tính. Tuy nhiên, chúng tôi nhắm mục tiêu rõ ràng vào việc phân loại theo luồng (streaming classification) thay vì đưa ra một dự đoán duy nhất cho mỗi chuỗi token. Hàm mất mát của chúng tôi cũng áp dụng việc đánh trọng số cho các tổn thất trên từng token (per-token losses), thay vì cho các logit, điều này có nghĩa là các hàm mất mát này không tương đương nhau. Hơn nữa, McKenzie và cộng sự (2025) không sử dụng kỹ thuật làm mượt logit (logit smoothing).

# 5.2 Phân tích

Chúng tôi phân tích tác động của các lựa chọn thiết kế probe khác nhau lên hiệu năng của bộ phân loại, đồng thời so sánh hiệu năng của linear probe với các bộ phân loại được fine-tune riêng biệt.

## Chi tiết huấn luyện

Chi tiết huấn luyện. Chúng tôi huấn luyện một bộ dò tuyến tính trên nền tảng của Claude Sonnet 4 bằng cách sử dụng các kích hoạt (activations) được ghép nối từ tất cả các lớp. 

Đối với tập dữ liệu trao đổi (exchange dataset), chúng tôi sử dụng cùng một dữ liệu huấn luyện như trong phần trước nhưng thay thế các nhãn cứng (hard labels) của quy trình tổng hợp bằng các nhãn mềm (soft labels) ở cấp độ trao đổi được tạo ra bởi một Bộ phân loại Hiến pháp (Constitutional Classifier) được tinh chỉnh từ Claude Sonnet 4.5. 

Để huấn luyện bộ dò, chúng tôi sử dụng nhiệt độ softmax $\tau = 1$, kích thước cửa sổ $M = 16$, và huấn luyện trên khoảng 25% toàn bộ tập dữ liệu trao đổi; chúng tôi nhận thấy rằng việc huấn luyện trên nhiều ví dụ hơn không giúp cải thiện hiệu suất của bộ dò. 

Để đánh giá, chúng tôi tính toán các dự đoán bằng cách sử dụng trung bình động có trọng số mũ (exponentially weighted moving average). Để làm tiêu chuẩn so sánh (baselines), chúng tôi so sánh với các bộ phân loại giai đoạn một và giai đoạn hai từ Phần 4. Tuy nhiên, trong phần này, cả hai bộ phân loại này đều được hiệu chuẩn ở tỷ lệ từ chối là $0,1\%$ trên WildChat (Zhao và cộng sự, 2024).

## Ensemble cải thiện độ bền vững

Chúng tôi thử nghiệm ensemble có trọng số giữa các bộ phân loại (Hình 3a). Với mỗi cặp bộ phân loại, chúng tôi kết hợp logit theo trung bình có trọng số:

$$
z_{ensemble} = \alpha z_1 + (1 - \alpha)z_2 
$$

trong đó $\alpha$ được tối ưu riêng cho từng cặp nhằm giảm tỷ lệ tấn công thành công trên tập kiểm tra.

Kết hợp probe với bộ phân loại nhỏ bên ngoài đạt tỷ lệ tấn công thành công thấp nhất, vượt trội so với từng bộ riêng lẻ. Đáng chú ý, trọng số bằng nhau ($\alpha = 0.5$) gần như tối ưu.

# C ADDITIONAL PROBE ABLATIONS

Hiện tại, chúng tôi đang tiến hành các nghiên cứu cắt bỏ sâu hơn để hiểu rõ hơn về tác động của các thiết kế hàm tổn thất và siêu tham số khác nhau đối với hiệu suất của đầu dò.

## C.1 Các hàm mất mát thay thế

Ngoài hàm mất mát softmax làm trơn được trình bày trong phần chính, chúng tôi khảo sát thêm hai công thức hàm mất mát tận dụng bản chất xử lý theo dòng (streaming) của bài toán phân loại.

### Hàm mất mát cực đại tích lũy

Chúng tôi thử nghiệm việc sử dụng **giá trị cực đại tích lũy** của các xác suất từ probe làm bộ dự đoán cho nhãn của toàn bộ chuỗi. Cách tiếp cận này phản ánh tốt hơn kịch bản sử dụng theo dòng, trong đó nếu ta dừng sinh tại vị trí $t$, phần còn lại của chuỗi được xem như có hại.

Cụ thể, ta định nghĩa:

$$
p(y = 1 \mid x_{1:t}) = \max_{\tau \le t} \sigma\big(\tilde{z}_\tau(x_{1:\tau})\big)
$$

Trong đó $\tilde{z}_\tau$ là logit của probe tại vị trí $\tau$ (sau khi làm trơn bằng cửa sổ trượt). Công thức này đảm bảo rằng xác suất dự đoán chỉ tăng khi probe tự tin rằng **toàn bộ chuỗi** sẽ là độc hại. Tuy nhiên, với cách làm này, gradient chỉ chảy qua vị trí token có điểm cao nhất, điều này đôi khi gây khó khăn cho quá trình tối ưu.

### Hàm mất mát cực đại tích lũy có ủ (annealed)

Để giải quyết khó khăn tối ưu của phương pháp cực đại tích lũy trong khi vẫn giữ được lợi ích cho phân loại theo dòng, chúng tôi nội suy tuyến tính giữa xác suất trực tiếp của probe và giá trị cực đại tích lũy trong suốt quá trình huấn luyện, theo đề xuất của **Sharma và cộng sự (2025)**:

$$
p(y = 1 \mid x_{1:t}) =
(1 - \omega)\,\sigma\big(\tilde{z}_t(x_{1:t})\big)
+ \omega \max_{\tau \le t} \sigma\big(\tilde{z}_\tau(x_{1:\tau})\big)
$$

Trong đó $\omega$ bắt đầu từ 0 và tăng tuyến tính lên 1 trong suốt quá trình huấn luyện. Cách này giúp giai đoạn đầu huấn luyện ổn định với luồng gradient tốt, sau đó dần chuyển sang công thức cực đại tích lũy phù hợp hơn với kịch bản sử dụng theo dòng.

**Hình 5a** so sánh các hàm mất mát này với phương pháp softmax làm trơn mà chúng tôi đề xuất. Kết quả cho thấy cách gán trọng số softmax kết hợp làm trơn activation của chúng tôi đạt hiệu năng tốt hơn các hàm mất mát bổ sung này. Chúng tôi sử dụng cùng bộ dữ liệu đánh giá và phương pháp như mô tả trong bài báo chính.

# D. Ensemble Weighting Analysis

Hình 6 xem xét ba cặp bộ phân loại và cho thấy hiệu suất riêng lẻ của chúng thay đổi như thế nào với các tỷ lệ đánh trọng số khác nhau. Chúng tôi sử dụng mức trung bình có trọng số của các logit cho mỗi cặp bộ phân loại. Trong khi chúng tôi tối ưu hóa hệ số $\alpha$ cho mỗi cặp bộ phân loại trong Hình 3a, thì đối với các tổ hợp bộ phân loại - bộ dò (classifier-probe ensembles), chúng tôi nhận thấy rằng việc đánh trọng số bằng nhau (equal weighting) đã gần đạt mức tối ưu.

# E. Tính toán kích hoạt tức thời (On-the-fly activation computation) 

Giúp tăng tốc phát triển bộ dò (probe). Trong quá trình thử nghiệm bộ phân loại bằng bộ dò, chúng tôi nhận thấy rằng việc tính toán lại các kích hoạt của mô hình ngay trong vòng lặp huấn luyện, thay vì tính toán trước rồi lưu trữ chúng, mang lại những lợi thế kỹ thuật đáng kể. Việc di chuyển dữ liệu kích hoạt của bộ dò từ bộ nhớ băng thông cao (HBM) sang RAM hoặc lưu trữ blob tạo ra các điểm nghẽn I/O nghiêm trọng, vượt xa cả chi phí tính toán của việc tái tạo (regeneration). Hơn nữa, vì việc huấn luyện bộ dò tuyến tính cực kỳ hiệu quả, chúng tôi có thể thử nghiệm đồng thời nhiều biến thể bộ dò trên các dữ liệu kích hoạt vừa được tính toán xong.