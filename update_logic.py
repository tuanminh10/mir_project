import sys

with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv2.py', 'r') as f:
    main_lines = f.readlines()

with open('temp_restore.txt', 'r') as f:
    restore_lines = f.readlines()

# Lọc bỏ index được in ra từ grep -n
clean_restore_lines = []
for line in restore_lines[:176]: # Cắt từ 0 đến trước def execute_task
    clean_restore_lines.append(line.split("-", 1)[1] if "-" in line[:5] else line.split(":", 1)[1])

start_idx = -1
end_idx = -1
for i, line in enumerate(main_lines):
    if "def verify_items_with_ai" in line:
        start_idx = i
    if "def execute_task" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_lines = main_lines[:start_idx] + clean_restore_lines + ["\n"] + main_lines[end_idx:]
    with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv2.py', 'w') as f:
        f.writelines(new_lines)
    print("Khôi phục hàm AI thành công")
