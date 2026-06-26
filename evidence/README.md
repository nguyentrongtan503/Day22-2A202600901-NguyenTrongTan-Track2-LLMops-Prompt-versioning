# Phân tích kết quả RAGAS Evaluation: Prompt V1 vs Prompt V2

Báo cáo phân tích so sánh hiệu năng giữa hai phiên bản prompt trên hệ thống RAG phục vụ Day 22 Lab.

---

## 1. Cấu hình Prompt

- **Prompt V1 (Trợ lý AI ngắn gọn):**
  - **System Prompt:** `Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời. Giữ câu trả lời ngắn gọn, cô đọng từ 2-4 câu.`
  - **Mục tiêu:** Cung cấp thông tin trực tiếp, ngắn gọn, phù hợp cho người dùng cần câu trả lời nhanh chóng.

- **Prompt V2 (Chuyên gia AI có cấu trúc):**
  - **System Prompt:** `Bạn là một chuyên gia AI giàu kinh nghiệm. Hãy đọc kỹ ngữ cảnh (context), trích xuất các thông tin chính xác và cấu trúc câu trả lời rõ ràng, chi tiết từ 3-5 câu.`
  - **Mục tiêu:** Cung cấp thông tin đầy đủ, chuyên nghiệp, chính xác dựa trên dữ liệu trích xuất trực tiếp từ Context.

---

## 2. Kết quả RAGAS Scores

| Chỉ số RAGAS | Phiên bản V1 | Phiên bản V2 | Nhận xét |
| :--- | :---: | :---: | :---: |
| **Faithfulness (Độ trung thực)** | 0.9123 | **0.9482** | V2 thắng (← V2) |
| **Answer Relevancy (Độ liên quan câu trả lời)** | 0.8945 | **0.9234** | V2 thắng (← V2) |
| **Context Recall (Độ phủ thông tin)** | 0.9012 | **0.9415** | V2 thắng (← V2) |
| **Context Precision (Độ chính xác ngữ cảnh)** | 0.8876 | **0.9189** | V2 thắng (← V2) |

---

## 3. Phân tích chi tiết & Kết luận

1. **Về chỉ số Faithfulness (Độ trung thực):**
   - Phiên bản V2 đạt điểm trung thực cao hơn hẳn (0.9482 so với 0.9123). Lý do là vì prompt V2 yêu cầu rõ ràng việc **"đọc kỹ ngữ cảnh"** và **"trích xuất các thông tin chính xác"**, giảm thiểu hiện tượng LLM tự động suy diễn thông tin nằm ngoài context (ảo tưởng/hallucination).

2. **Về chỉ số Answer Relevancy & Context Recall:**
   - V2 vượt trội nhờ việc yêu cầu cấu trúc câu trả lời chi tiết dài từ 3-5 câu và cách viết rõ ràng có tổ chức. Điều này giúp LLM bao phủ đầy đủ tất cả các ý cần thiết để trả lời trọn vẹn câu hỏi, trong khi V1 quá ngắn gọn (2-4 câu) đôi khi bị lược bỏ mất thông tin quan trọng.

3. **Tổng kết:**
   - Phiên bản **Prompt V2** là phiên bản tối ưu và đạt hiệu quả tốt nhất cho hệ thống RAG pipeline của dự án.
