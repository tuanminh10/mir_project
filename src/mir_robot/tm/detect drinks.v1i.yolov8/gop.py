import os
import glob

# Lấy thư mục chứa script để tạo đường dẫn động
base_dir = os.path.dirname(os.path.abspath(__file__))

label_dirs = [
   os.path.join(base_dir, 'train', 'labels'),
   os.path.join(base_dir, 'valid', 'labels'),
   os.path.join(base_dir, 'test', 'labels')
]


class_mapping = {
   0: 0,  # Coca-Cola -> Coca
   1: 0,  # coca-cola -> Coca
   2: 0,  # cocacola -> Coca
   3: 1,  # water_bottle -> Lavie
   4: 1   # waterbottle -> Lavie
}


def process_labels():
   for label_dir in label_dirs:
       if not os.path.exists(label_dir):
           print(f"⚠️ Bỏ qua: Không tìm thấy thư mục {label_dir}")
           continue
          
       txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
       print(f"⏳ Đang phẫu thuật {len(txt_files)} file trong {label_dir}...")
      
       count_modified = 0
       for file_path in txt_files:
           with open(file_path, 'r') as f:
               lines = f.readlines()
              
           new_lines = []
           for line in lines:
               parts = line.strip().split()
               if len(parts) == 5:
                   old_class_id = int(parts[0])
                   if old_class_id in class_mapping:
                       new_class_id = class_mapping[old_class_id]
                       parts[0] = str(new_class_id)
                       new_lines.append(' '.join(parts) + '\n')
          
           with open(file_path, 'w') as f:
               f.writelines(new_lines)
           count_modified += 1
              
       print(f"  -> Đã xử lý xong {count_modified} file.\n")
      
   print("✅ XUẤT SẮC! Toàn bộ Dataset đã được gộp thành 2 class!")


if __name__ == '__main__':
   process_labels()
