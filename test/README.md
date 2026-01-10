# Test Scripts for Step Boundaries Detection

Các script test để kiểm tra việc detect step boundaries trong Step-JEPA training.

## Files

1. **test_step_boundaries_simple.py**: Test detect `\n\n` trong raw text (không cần transformers)
   - Kiểm tra xem các vị trí `\n\n` đầu tiên và thứ hai có khác nhau giữa các samples không
   - Chạy: `python3 test_step_boundaries_simple.py`

2. **test_step_boundaries.py**: Test detect `\n\n` sau khi tokenize (cần transformers)
   - Kiểm tra xem sau khi tokenize, các vị trí có còn khác nhau không
   - Kiểm tra xem `\n\n` đầu tiên có nằm trong phần system/user không
   - Chạy: `python3 test_step_boundaries.py`

3. **test_tokenized_boundaries.md**: Hướng dẫn và script để test sau khi tokenize
   - Script Python để chạy trong notebook hoặc môi trường có transformers
   - Kiểm tra chi tiết vị trí step boundaries sau tokenization

## Vấn đề đang điều tra

Khi training, tất cả samples trong batch có cùng `index_predictor` và `index_step2`, dẫn đến cùng `jepa_loss`. Cần kiểm tra:

1. `\n\n` đầu tiên có nằm trong phần system/user message không?
2. Sau khi tokenize, các vị trí có bị align lại không?
3. Có phải do padding làm cho các vị trí giống nhau không?



